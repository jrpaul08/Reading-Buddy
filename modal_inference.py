import os

import modal

app = modal.App("reading-buddy")

MODEL_ID = "openbmb/MiniCPM-o-4_5"
MODEL_DIR = "/model-weights"

vol = modal.Volume.from_name("reading-buddy-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "torchvision",
        "torchcodec",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.51.0",
        "accelerate",
        "minicpmo-utils[all]>=1.0.5",
        "einops",
        "bitsandbytes",
        "huggingface_hub",
        "numpy",
        "sentencepiece",
        "Pillow",
        "soundfile",
        "librosa",
    )
    .add_local_file("narrator_ref.wav", "/narrator_ref.wav")
)

NARRATOR_REF_PATH = "/narrator_ref.wav"


@app.cls(
    gpu="A100-40GB",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: vol},
    scaledown_window=600,
    timeout=900,
)
class ReadingCompanion:

    @modal.enter()
    def enter(self):
        """
        Purpose: Runs once when the Modal container starts. Downloads model weights
        to the persistent Volume if not already present, then loads the model and
        tokenizer into GPU memory. All subsequent method calls on this container
        reuse the already-loaded model.

        Args:
            None

        Returns:
            None — sets self.model and self.tokenizer as instance attributes.
        """
        import time
        import torch
        import torchaudio
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        # stepaudio2's Token2wav writes synthesized audio via torchaudio.save() to an
        # in-memory BytesIO buffer, but the installed torchaudio's torchcodec backend
        # only supports writing to file paths, not file-like objects. Patch save() to
        # fall back to soundfile (which our image already depends on) for that case.
        _original_torchaudio_save = torchaudio.save

        def _patched_save(uri, src, sample_rate, *args, **kwargs):
            if hasattr(uri, "write"):
                arr = src.cpu().numpy().T
                if arr.shape[1] == 1:
                    arr = arr[:, 0]
                sf.write(uri, arr, sample_rate, format="WAV")
                return
            return _original_torchaudio_save(uri, src, sample_rate, *args, **kwargs)

        torchaudio.save = _patched_save

        vol.reload()

        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_DIR,
                token=os.environ["HF_TOKEN"],
            )
            vol.commit()

        t0 = time.time()
        self.model = AutoModel.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().cuda()
        self.model.init_tts()
        print(f"[model load] {time.time() - t0:.1f}s")

        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
        )
        print(f"[tokenizer load] {time.time() - t1:.1f}s")

    @modal.method()
    def answer_from_audio(self, audio_bytes: bytes) -> dict:
        """
        Purpose: Accepts raw audio bytes, transcribes the speaker's question, then
        answers it. Two separate model calls are made so the transcription can be
        verified independently — this is important because a bad transcription
        will produce a bad answer regardless of model quality.

        Args:
            audio_bytes (bytes): Raw audio file content (WAV, M4A, MP3, etc.).

        Returns:
            dict with two keys:
                "question" (str): The model's transcription of what the speaker said.
                "answer"   (str): The model's answer to the transcribed question.
        """
        import io
        import time
        import torch
        import numpy as np
        import soundfile as sf

        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        audio_array = audio_array.astype(np.float32)

        t0 = time.time()
        with torch.inference_mode():
            transcription = self.model.chat(
                msgs=[{"role": "user", "content": [audio_array, "Transcribe exactly what the speaker is saying."]}],
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )
        print(f"[transcription] {time.time() - t0:.1f}s: {transcription}")

        t1 = time.time()
        with torch.inference_mode():
            answer = self.model.chat(
                msgs=[{"role": "user", "content": f"Answer this question: {transcription}"}],
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )
        print(f"[answer] {time.time() - t1:.1f}s")

        return {"question": transcription, "answer": answer}

    @modal.method()
    def answer_spoken(self, audio_bytes: bytes) -> dict:
        """
        Purpose: Accepts raw audio bytes, transcribes the speaker's question, generates
        a text answer, then converts that answer to speech using MiniCPM-o's zero-shot
        TTS (voice cloning from a reference audio clip).

        Args:
            audio_bytes (bytes): Raw audio file content (WAV, M4A, MP3, etc.).

        Returns:
            dict with three keys:
                "question"     (str):   The model's transcription of what the speaker said.
                "answer_text"  (str):   The model's answer as plain text.
                "answer_audio" (bytes): WAV audio bytes of the spoken answer.
        """
        import io
        import time
        import torch
        import numpy as np
        import librosa
        import soundfile as sf

        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        audio_array = audio_array.astype(np.float32)

        t0 = time.time()
        with torch.inference_mode():
            transcription = self.model.chat(
                msgs=[{"role": "user", "content": [audio_array, "Transcribe exactly what the speaker is saying."]}],
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )
        print(f"[transcription] {time.time() - t0:.1f}s: {transcription}")

        t1 = time.time()
        with torch.inference_mode():
            answer_text = self.model.chat(
                msgs=[{"role": "user", "content": (
                    f"Answer this question in 3-5 sentences with helpful detail: {transcription}"
                )}],
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )
        print(f"[text answer] {time.time() - t1:.1f}s: {answer_text}")

        # Reference audio determines the cloned voice for the spoken response.
        # Use the project's narrator reference clip if present, else fall back to
        # the speaker's own (resampled) voice.
        if os.path.exists(NARRATOR_REF_PATH):
            ref_audio, _ = librosa.load(NARRATOR_REF_PATH, sr=16000, mono=True)
        else:
            ref_audio = audio_array

        sys_msg = {
            "role": "system",
            "content": [
                "Clone the voice in the provided audio prompt.",
                ref_audio,
                "Please read the following text aloud directly, with no additional commentary.",
            ],
        }
        user_msg = {"role": "user", "content": [answer_text]}

        output_audio_path = "/tmp/response.wav"
        t2 = time.time()
        with torch.inference_mode():
            self.model.chat(
                msgs=[sys_msg, user_msg],
                tokenizer=self.tokenizer,
                do_sample=True,
                max_new_tokens=512,
                use_tts_template=True,
                generate_audio=True,
                temperature=0.1,
                output_audio_path=output_audio_path,
            )
        print(f"[tts] {time.time() - t2:.1f}s")

        with open(output_audio_path, "rb") as f:
            audio_wav_bytes = f.read()

        return {"question": transcription, "answer_text": answer_text, "answer_audio": audio_wav_bytes}

    @modal.method()
    def generate_text(self, prompt: str) -> str:
        """
        Purpose: Generates a text response from a plain text prompt. Used for
        testing the model directly or as a fallback when no audio input is present.

        Args:
            prompt (str): The user's question or instruction as plain text.

        Returns:
            str: The model's text response.
        """
        import time
        import torch

        t0 = time.time()
        with torch.inference_mode():
            response = self.model.chat(
                msgs=[{"role": "user", "content": prompt}],
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )
        print(f"[inference] {time.time() - t0:.1f}s")
        return response


@app.local_entrypoint()
def main():
    """
    Purpose: Local test entrypoint for the spoken response pipeline.
    Reads voice-prompt1.wav from the project root, sends it to answer_spoken,
    prints the transcribed question and text answer, and saves the spoken
    audio response to response.wav.

    Args:
        None

    Returns:
        None — results are printed to stdout and audio is saved to response.wav.

    Usage:
        modal run modal_inference.py
    """
    with open("voice-prompt1.wav", "rb") as f:
        audio_bytes = f.read()

    companion = ReadingCompanion()
    result = companion.answer_spoken.remote(audio_bytes)

    print(f"\nQuestion heard: {result['question']}")
    print(f"\nAnswer: {result['answer_text']}")

    with open("response.wav", "wb") as f:
        f.write(result["answer_audio"])
    print("\nAudio response saved to response.wav")
