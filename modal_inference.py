import os

import modal

from prompt_utils import ANSWER_GENERATION_KWARGS, ANSWER_INSTRUCTION_SUFFIX, build_system_prompt

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
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("voice-prompts/narrator_ref.wav", "/narrator_ref.wav")
    .add_local_file("book_utils.py", "/root/book_utils.py")
    .add_local_file("prompt_utils.py", "/root/prompt_utils.py")
    .add_local_dir("books", "/root/books")
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

    def _transcribe(self, audio_bytes: bytes):
        """
        Purpose: Decode raw audio bytes to a mono 16kHz float32 array and
        transcribe the speaker's question.

        Args:
            audio_bytes (bytes): Raw audio file content (WAV, M4A, MP3, etc.).

        Returns:
            tuple[str, np.ndarray]:
                transcription (str): The model's transcription of what the speaker said.
                audio_array (np.ndarray): The decoded mono 16kHz audio.
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

        return transcription, audio_array

    @modal.method()
    def answer_spoken(self, audio_bytes: bytes, book_name: str = None, chapter_number: int = None) -> dict:
        """
        Purpose: Accepts raw audio bytes, transcribes the speaker's question, generates
        a text answer, then converts that answer to speech using MiniCPM-o's zero-shot
        TTS (voice cloning from a reference audio clip).

        Args:
            audio_bytes (bytes): Raw audio file content (WAV, M4A, MP3, etc.).
            book_name (str, optional): Book identifier (e.g. "crime_and_punishment").
                If provided along with chapter_number, the answer is grounded in the
                book's text up to that chapter using a spoiler-free system prompt.
            chapter_number (int, optional): The reader's current chapter (1-indexed).
                Required if book_name is provided.

        Returns:
            dict with three keys:
                "question"     (str):   The model's transcription of what the speaker said.
                "answer_text"  (str):   The model's answer as plain text.
                "answer_audio" (bytes): WAV audio bytes of the spoken answer.
        """
        import time
        import torch
        import librosa

        transcription, audio_array = self._transcribe(audio_bytes)

        if book_name is not None:
            from book_utils import describe_reading_context
            context, source_label = describe_reading_context(book_name, chapter_number)
            answer_msgs = [
                {"role": "system", "content": build_system_prompt(context, source_label)},
                {"role": "user", "content": f"{transcription}{ANSWER_INSTRUCTION_SUFFIX}"},
            ]
        else:
            answer_msgs = [{"role": "user", "content": (
                f"Answer this question in 3-5 sentences with helpful detail: {transcription}"
            )}]

        t1 = time.time()
        with torch.inference_mode():
            answer_text = self.model.chat(
                msgs=answer_msgs,
                tokenizer=self.tokenizer,
                **ANSWER_GENERATION_KWARGS,
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
    def answer_text_questions(self, context: str, questions: list, source_label: str) -> list:
        """
        Purpose: Text-only batch testing helper. Given a hand-written context block
        (e.g. a chapter summary or structured data) and a list of questions, runs
        each question through the same prompt structure and decoding settings as
        answer_spoken's text-answer step, with no audio transcription or TTS. Used
        to compare alternative context representations (summaries, structured data)
        against the standard chapter-text context.

        Args:
            context (str): Hand-written context text (e.g. a summary of recent
                chapters, or a structured block of characters/events).
            questions (list[str]): Questions to ask, each evaluated independently
                (no shared conversation history between questions).
            source_label (str): Short description of the context's source, used
                in the prompt header (e.g. "a summary of chapters 1-2 of Crime
                and Punishment").

        Returns:
            list[dict]: One dict per question, each with keys:
                "question" (str): The input question.
                "answer"   (str): The model's text answer.
        """
        import time
        import torch

        system_prompt = build_system_prompt(context, source_label)

        results = []
        for question in questions:
            answer_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{question}{ANSWER_INSTRUCTION_SUFFIX}"},
            ]

            t0 = time.time()
            with torch.inference_mode():
                answer = self.model.chat(
                    msgs=answer_msgs,
                    tokenizer=self.tokenizer,
                    **ANSWER_GENERATION_KWARGS,
                )
            print(f"[{time.time() - t0:.1f}s] Q: {question}\nA: {answer}\n")
            results.append({"question": question, "answer": answer})

        return results
