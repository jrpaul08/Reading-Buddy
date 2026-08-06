import os

import modal
from fastapi import File, Form, UploadFile

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
        "fastapi",
        "python-multipart",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("voice-prompts/narrator-voice.wav", "/narrator-voice.wav")
    .add_local_file("book_utils.py", "/root/book_utils.py")
    .add_local_file("omni/prompt_utils.py", "/root/prompt_utils.py")
    .add_local_dir("books", "/root/books")
)

NARRATOR_REF_PATH = "/narrator-voice.wav"


@app.cls(
    gpu="H100",
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

    def _transcribe(self, audio_bytes: bytes) -> str:
        """
        Purpose: Decode raw audio bytes to a mono 16kHz float32 array and
        transcribe the speaker's question.

        Args:
            audio_bytes (bytes): Raw audio file content (WAV, M4A, MP3, etc.).

        Returns:
            str: The model's transcription of what the speaker said.
        """
        import io
        import subprocess
        import time
        import torch
        import numpy as np
        import librosa
        import soundfile as sf

        # Browser MediaRecorder uploads arrive as WebM (EBML container,
        # magic bytes 0x1A45DFA3), which soundfile can't read. Transcode to
        # WAV via ffmpeg first; WAV/other formats soundfile already supports
        # are passed through unchanged.
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            audio_bytes = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                input=audio_bytes,
                capture_output=True,
                check=True,
            ).stdout

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

        return transcription

    @modal.method()
    def run_s2s_pipeline(self, audio_bytes: bytes, book_name: str, chapter_number: int = None, chapter_numbers: list = None) -> dict:
        """
        Purpose: Accepts raw audio bytes, transcribes the speaker's question, generates
        a book-grounded text answer using a spoiler-free system prompt, then converts
        that answer to speech using MiniCPM-o's zero-shot TTS (voice cloning from a
        reference audio clip).

        Args:
            audio_bytes (bytes): Raw audio file content (WAV, M4A, MP3, etc.).
            book_name (str): Book identifier (e.g. "crime_and_punishment").
            chapter_number (int, optional): The reader's current chapter (1-indexed).
                If set (and chapter_numbers is not), uses the raw chapter text up to
                this chapter via describe_reading_context.
            chapter_numbers (list[int], optional): Chapter numbers to include via
                describe_hybrid_context (book_chapter_context.json summaries/structured
                data), e.g. [1, 2]. Takes precedence over chapter_number if both are set.

        Returns:
            dict with three keys:
                "question"     (str):   The model's transcription of what the speaker said.
                "answer_text"  (str):   The model's answer as plain text.
                "answer_audio" (bytes): WAV audio bytes of the spoken answer.
        """
        import time
        import torch
        import librosa

        transcription = self._transcribe(audio_bytes)

        if chapter_numbers is not None:
            from book_utils import describe_hybrid_context
            context, source_label = describe_hybrid_context(book_name, chapter_numbers)
        else:
            from book_utils import describe_reading_context
            context, source_label = describe_reading_context(book_name, chapter_number)

        print(f"[run_s2s_pipeline] chapter_number={chapter_number!r} chapter_numbers={chapter_numbers!r} context length: {len(context)} chars")

        answer_msgs = [
            {"role": "system", "content": build_system_prompt(context, source_label)},
            {"role": "user", "content": f"{transcription}{ANSWER_INSTRUCTION_SUFFIX}"},
        ]

        t1 = time.time()
        with torch.inference_mode():
            answer_text = self.model.chat(
                msgs=answer_msgs,
                tokenizer=self.tokenizer,
                **ANSWER_GENERATION_KWARGS,
            )
        print(f"[text answer] {time.time() - t1:.1f}s: {answer_text}")

        # Reference audio determines the cloned voice for the spoken response.
        ref_audio, _ = librosa.load(NARRATOR_REF_PATH, sr=16000, mono=True)

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

    @modal.fastapi_endpoint(method="POST")
    async def s2s_endpoint(
        self,
        audio: UploadFile = File(...),
        book_id: str = Form(...),
        book_title: str = Form(...),
        author: str = Form(...),
        chapter: int = Form(...),
    ):
        """
        Purpose: Public HTTP endpoint wrapping run_s2s_pipeline. Accepts a
        multipart/form-data POST with an audio file and book context fields,
        runs the full speech-to-speech pipeline, and returns the spoken
        answer as a WAV file.

        Args:
            audio (UploadFile): Uploaded audio file (the listener's question).
            book_id (str): Book identifier (e.g. "crime_and_punishment"),
                matching a key in book_utils.AVAILABLE_BOOKS.
            book_title (str): Book title. Not used by the pipeline (derived
                from book_id internally) — accepted for frontend convenience.
            author (str): Book author. Not used by the pipeline — accepted
                for frontend convenience.
            chapter (int): The reader's current chapter (1-indexed).

        Returns:
            fastapi.Response: WAV audio bytes of the spoken answer, with
            media type "audio/wav". On invalid book_id/chapter, returns a
            400 JSON error response instead.
        """
        from fastapi import Response
        from fastapi.responses import JSONResponse

        print(f"[s2s_endpoint] audio: {audio!r} (filename={audio.filename!r}, content_type={audio.content_type!r})")
        print(f"[s2s_endpoint] book_id: {book_id!r} (type={type(book_id)})")
        print(f"[s2s_endpoint] book_title: {book_title!r} (type={type(book_title)})")
        print(f"[s2s_endpoint] author: {author!r} (type={type(author)})")
        print(f"[s2s_endpoint] chapter: {chapter!r} (type={type(chapter)})")

        # book_title/author are accepted but unused — run_s2s_pipeline derives
        # both from book_id via book_utils.AVAILABLE_BOOKS.
        del book_title, author

        from book_utils import describe_hybrid_context

        chapter_numbers = list(range(1, chapter + 1))
        context, source_label = describe_hybrid_context(book_id, chapter_numbers)
        system_prompt = build_system_prompt(context, source_label)

        import datetime

        debug_log_path = os.path.join(MODEL_DIR, "s2s_debug.log")
        with open(debug_log_path, "a") as f:
            f.write(f"\n===== {datetime.datetime.now().isoformat()} =====\n")
            f.write(f"book_id: {book_id!r}\n")
            f.write(f"chapter: {chapter!r}\n")
            f.write(f"context length: {len(context)} chars\n")
            f.write(f"context head (first 500 chars):\n{context[:500]!r}\n")
            f.write(f"context tail (last 500 chars):\n{context[-500:]!r}\n")
            f.write(f"system prompt:\n{system_prompt}\n")
        vol.commit()

        audio_bytes = await audio.read()

        try:
            result = self.run_s2s_pipeline.local(
                audio_bytes, book_name=book_id, chapter_numbers=chapter_numbers
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        return Response(content=result["answer_audio"], media_type="audio/wav")

    @modal.method()
    def answer_text_questions(self, context: str, questions: list, source_label: str) -> list:
        """
        Purpose: Text-only batch testing helper. Given a hand-written context block
        (e.g. a chapter summary or structured data) and a list of questions, runs
        each question through the same prompt structure and decoding settings as
        run_s2s_pipeline's text-answer step, with no audio transcription or TTS. Used
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
