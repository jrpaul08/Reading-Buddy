"""
Orchestrator for the Reading Buddy three-model pipeline. A CPU-only Modal
class (no GPU, so near-zero idle cost) that chains the three GPU
components behind the same public contract the omni model's
s2s_endpoint exposes, so the frontend doesn't change:

    audio in  ->  STTEngine.transcribe      (question text)
              ->  ReasoningEngine.answer    (answer text, book-grounded)
              ->  TTSEngine.synthesize      (WAV bytes)  ->  audio out

Book context is built here, not in the reasoning container, so the
books data only needs to be mounted in this one place.

Deployed under the "reading-buddy-pipeline" app name while under
development, which gives it its own URLs and leaves the live omni app
untouched. Cutting over later means pointing the frontend's URL secrets
at this app's endpoints.

Built up incrementally:
    Part 1 (current scope): file header, image, and the class shell.
    Part 2: run_pipeline — the three stages chained, with per-stage timing.
    Part 3: s2s_endpoint — the public HTTP wrapper around run_pipeline.
    Part 4: warmup_endpoint — wakes all three GPU containers in parallel.
"""

import modal

from modal_app import app

# Importing the component classes is what registers them on the shared
# app, so deploying this file ships all four classes together.
from reasoning_modal import ReasoningEngine
from stt_modal import STTEngine
from tts_modal import TTSEngine

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Needed for the public endpoint's request/response types and for
        # parsing multipart form uploads (the audio file).
        "fastapi",
        "python-multipart",
    )
    # Importing the three component files (above) pulls in modal_app and,
    # via reasoning_modal, prompt_utils, so those must exist in the
    # container too. Derived by tracing each file's module-level imports,
    # not guessed.
    .add_local_file("pipeline/modal_app.py", "/root/modal_app.py")
    .add_local_file("pipeline/reasoning_modal.py", "/root/reasoning_modal.py")
    .add_local_file("pipeline/stt_modal.py", "/root/stt_modal.py")
    .add_local_file("pipeline/tts_modal.py", "/root/tts_modal.py")
    .add_local_file("pipeline/prompt_utils.py", "/root/prompt_utils.py")
    # Book context is built here, at request time.
    .add_local_file("book_utils.py", "/root/book_utils.py")
    .add_local_dir("books", "/root/books")
)


@app.cls(
    image=image,
    scaledown_window=600,
    timeout=900,
)
class Orchestrator:
    """
    Chains STTEngine -> ReasoningEngine -> TTSEngine and exposes the
    result over HTTP. CPU-only: it holds no models, only routes bytes and
    text between the GPU components and builds the book context.
    """

    @modal.method()
    def run_pipeline(self, audio_bytes: bytes, book_id: str, chapter: int) -> dict:
        """
        Purpose: Runs the full speech-to-speech pipeline: transcribes the
        reader's spoken question, answers it from the book (spoiler-safe:
        only chapters 1 through `chapter` are ever loaded), and speaks the
        answer. A plain method, not an HTTP endpoint, so it can be tested
        with `modal run` and inspected stage by stage; the public endpoint
        is a thin wrapper around this.

        Args:
            audio_bytes (bytes): The recorded question, in any format the
                STT component accepts (WAV, WebM/Opus, ...).
            book_id (str): Book identifier, e.g. "crime_and_punishment".
            chapter (int): The reader's current chapter (1-indexed).
                Context is built from chapters 1 through this one.

        Returns:
            dict with keys:
                "question"     (str):   What the reader said, transcribed.
                "answer_text"  (str):   The book-grounded answer.
                "answer_audio" (bytes): WAV audio of the spoken answer.
                "timings"      (dict):  Seconds spent in each stage
                    ("stt", "reasoning", "tts", "total"), for tracking
                    latency.

        Raises:
            ValueError: For an unknown book_id, or if the TTS component
                produces no audio for the answer.
        """
        import time

        t_start = time.time()

        t0 = time.time()
        question = STTEngine().transcribe.remote(audio_bytes)
        stt_time = time.time() - t0
        print(f"[stt] {stt_time:.1f}s: {question!r}")

        from book_utils import describe_hybrid_context

        chapter_numbers = list(range(1, chapter + 1))
        context, source_label = describe_hybrid_context(book_id, chapter_numbers)
        print(f"[context] {len(context)} chars from chapters 1-{chapter}")

        t0 = time.time()
        results = ReasoningEngine().answer.remote(
            context=context,
            questions=[question],
            source_label=source_label,
        )
        answer_text = results[0]["answer"]
        reasoning_time = time.time() - t0
        print(f"[reasoning] {reasoning_time:.1f}s: {answer_text!r}")

        t0 = time.time()
        answer_audio = TTSEngine().synthesize.remote(answer_text)
        tts_time = time.time() - t0
        print(f"[tts] {tts_time:.1f}s: {len(answer_audio)} bytes of audio")

        total_time = time.time() - t_start
        print(f"[total] {total_time:.1f}s")

        return {
            "question": question,
            "answer_text": answer_text,
            "answer_audio": answer_audio,
            "timings": {
                "stt": stt_time,
                "reasoning": reasoning_time,
                "tts": tts_time,
                "total": total_time,
            },
        }
