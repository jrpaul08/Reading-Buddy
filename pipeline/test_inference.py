"""
Debug/experimental entrypoints for the Reading Buddy pipeline. Mirrors the
structure of omni/test_inference.py, built up incrementally alongside
each pipeline component (reasoning_modal.py, stt_modal.py, ...).

Usage:
    modal run pipeline/test_inference.py::check_load
    modal run pipeline/test_inference.py::ask_question
    modal run pipeline/test_inference.py::test_book
    modal run pipeline/test_inference.py::check_stt_load
    modal run pipeline/test_inference.py::test_transcribe
    modal run pipeline/test_inference.py::check_tts_load
    modal run pipeline/test_inference.py::test_synthesize
    modal run pipeline/test_inference.py::test_pipeline
"""

import json
import time
from pathlib import Path

from orchestrator_modal import Orchestrator
from reasoning_modal import app, ReasoningEngine
from stt_modal import STTEngine
from tts_modal import TTSEngine


@app.local_entrypoint()
def check_load():
    """
    Purpose: Part 1 sanity check. Spins up the ReasoningEngine container,
    which downloads and loads Qwen2.5-14B-Instruct, then reports back
    whether it landed on GPU in bfloat16 with the expected parameter
    count. No text generation yet — that's Part 2.

    Args:
        None

    Returns:
        None — result is printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::check_load
    """
    engine = ReasoningEngine()
    result = engine.ping.remote()

    print(f"\nmodel_loaded: {result['model_loaded']}")
    print(f"device:       {result['device']}")
    print(f"dtype:        {result['dtype']}")
    print(f"num_params:   {result['num_params']:,} ({result['num_params'] / 1e9:.1f}B)")


@app.local_entrypoint()
def ask_question(question: str = "What is the capital of France?"):
    """
    Purpose: Part 2 sanity check. Sends a plain question, with no book
    context or system prompt, straight to Qwen and prints its answer —
    proving the tokenize -> generate -> decode chain works before Part 3/4
    add real prompt structure and book-grounded context.

    Args:
        question (str): A plain question with a known/checkable answer.
            Defaults to "What is the capital of France?" — a factual,
            trivially-verifiable sanity check that has nothing to do with
            any book.

    Returns:
        None — the question and answer are printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::ask_question
        modal run pipeline/test_inference.py::ask_question --question "What is 7 times 8?"
    """
    engine = ReasoningEngine()
    answer = engine.ask.remote(question)

    print(f"\nQ: {question}")
    print(f"A: {answer}")


@app.local_entrypoint()
def test_book(
    book_name: str = "crime_and_punishment",
    chapter_number: int = 4,
    questions: str = "",
    questions_file: str = "",
):
    """
    Purpose: Part 4 sanity check — the first real, book-grounded test.
    Builds hybrid context (summary + structured data) for the given book
    up to the given chapter, then answers one or more questions using the
    real spoiler-prevention system prompt and generation settings. Mirrors
    omni/test_inference.py's test_text, so results can be compared
    directly against the omni model's answers to the same questions.

    Args:
        book_name (str): Book identifier, e.g. "crime_and_punishment".
        chapter_number (int): The reader's current chapter (1-indexed).
            Context includes chapters 1 through this chapter. Defaults to 4.
        questions (str): Questions separated by | (pipe), typed directly
            on the command line. Takes priority over questions_file if
            both are given.
        questions_file (str): Name of a file inside
            books/<book_name>/test_questions/ to load questions from,
            e.g. "test_question_ch21". Ignored if questions is given. If
            neither is given, falls back to test_questions.json in that
            same folder — the file omni/test_inference.py's
            test_context_qa also reads, so editing that one list
            improves test coverage for both models at once.

    Returns:
        None — results are printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::test_book
        modal run pipeline/test_inference.py::test_book --chapter-number 7 --questions "Who is Sonya?|What is the axe for?"
        modal run pipeline/test_inference.py::test_book --chapter-number 21 --questions-file test_question_ch21
    """
    from book_utils import describe_hybrid_context

    chapter_numbers = list(range(1, chapter_number + 1))
    context, source_label = describe_hybrid_context(book_name, chapter_numbers)

    questions_dir = Path(__file__).parent.parent / "books" / book_name / "test_questions"

    if questions:
        question_list = [q.strip() for q in questions.split("|") if q.strip()]
    elif questions_file:
        with open(questions_dir / questions_file) as f:
            question_list = json.load(f)
    else:
        with open(questions_dir / "test_questions.json") as f:
            question_list = json.load(f)

    engine = ReasoningEngine()
    results = engine.answer.remote(
        context=context,
        questions=question_list,
        source_label=source_label,
    )

    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")


@app.local_entrypoint()
def check_stt_load():
    """
    Purpose: Part 1 sanity check for the STT component. Spins up the
    STTEngine container, which downloads and loads Whisper large-v3-turbo
    via faster-whisper, then reports back the configuration it loaded
    with. No transcription yet — that's Part 2.

    Args:
        None

    Returns:
        None — result is printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::check_stt_load
    """
    engine = STTEngine()
    result = engine.ping.remote()

    print(f"\nmodel_loaded: {result['model_loaded']}")
    print(f"device:       {result['device']}")
    print(f"compute_type: {result['compute_type']}")


@app.local_entrypoint()
def test_transcribe(audio_file: str = "voice-prompts/voice-prompt-ch7.wav"):
    """
    Purpose: Reads a real local audio file and sends it to
    STTEngine.transcribe, printing the result. Works with WAV or
    WebM/Opus (verified against both) — any format faster-whisper's
    underlying PyAV decoder supports.

    Args:
        audio_file (str): Path to a local audio file (WAV or WebM).
            Defaults to one of the existing voice-prompts/ recordings
            used by the omni model's own tests.

    Returns:
        None — the transcription is printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::test_transcribe
        modal run pipeline/test_inference.py::test_transcribe --audio-file voice-prompts/voice-prompt-ch2.wav
    """
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    engine = STTEngine()
    transcription = engine.transcribe.remote(audio_bytes)

    print(f"\nTranscription: {transcription}")


@app.local_entrypoint()
def check_tts_load():
    """
    Purpose: Part 1 sanity check for the TTS component. Spins up the
    TTSEngine container, which loads Kokoro via KPipeline, then reports
    back the configuration it loaded with. No synthesis yet — that's
    Part 2.

    Args:
        None

    Returns:
        None — result is printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::check_tts_load
    """
    engine = TTSEngine()
    result = engine.ping.remote()

    print(f"\npipeline_loaded: {result['pipeline_loaded']}")
    print(f"device:          {result['device']}")
    print(f"lang_code:       {result['lang_code']}")


@app.local_entrypoint()
def test_synthesize(text: str = "The Protagonist of the story is Raskolnikov"):
    """
    Purpose: Part 2 sanity check. Sends real text to TTSEngine.synthesize
    and saves the resulting audio locally so it can actually be listened
    to, proving the KPipeline synthesis chain works with the chosen
    voice before Part 3 wires this into the orchestrator-ready shape.

    Args:
        text (str): The text to speak aloud. Defaults to a real answer
            from earlier testing of the reasoning component.

    Returns:
        None — audio is saved to response_tts.wav.

    Usage:
        modal run pipeline/test_inference.py::test_synthesize
        modal run pipeline/test_inference.py::test_synthesize --text "Who is Sonya?"
    """
    engine = TTSEngine()
    audio_bytes = engine.synthesize.remote(text)

    with open("response_tts.wav", "wb") as f:
        f.write(audio_bytes)

    print(f"\nText:  {text}")
    print("Audio saved to response_tts.wav")


@app.local_entrypoint()
def test_pipeline(
    audio_file: str = "voice-prompts/voice-prompt-ch7.wav",
    book_id: str = "crime_and_punishment",
    chapter: int = 7,
):
    """
    Purpose: End-to-end test of the full pipeline through the
    orchestrator, without HTTP. Sends a real recorded question through
    speech-to-text, book-grounded reasoning, and text-to-speech, then
    prints what was heard, the answer, and per-stage timings, and saves
    the spoken answer so it can be listened to. Mirrors omni's test_book.

    Args:
        audio_file (str): Path to a local recording of the question.
        book_id (str): Book identifier. Defaults to "crime_and_punishment".
        chapter (int): The reader's current chapter. Defaults to 7.

    Returns:
        None — results are printed, and audio is saved to
        response_pipeline.wav. The pipeline is called twice in one run:
        the first call includes every container's cold start, the second
        hits the now-warm containers. "wall-clock" is measured here, from
        audio sent to answer audio back, and is what omni's test reports
        too, so the two are directly comparable. The stage rows and
        "total" are measured inside the orchestrator.

    Usage:
        modal run pipeline/test_inference.py::test_pipeline
        modal run pipeline/test_inference.py::test_pipeline --chapter 2 --audio-file voice-prompts/voice-prompt-ch2.wav
    """
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    orchestrator = Orchestrator()

    t0 = time.time()
    cold = orchestrator.run_pipeline.remote(audio_bytes, book_id, chapter)
    cold_wall = time.time() - t0

    t0 = time.time()
    warm = orchestrator.run_pipeline.remote(audio_bytes, book_id, chapter)
    warm_wall = time.time() - t0

    print(f"\nQuestion heard: {cold['question']}")
    print(f"\nAnswer: {cold['answer_text']}")

    print("\nTimings (seconds)      cold     warm")
    for stage in ("stt", "reasoning", "tts", "total"):
        print(f"  {stage:<12} {cold['timings'][stage]:>8.1f} {warm['timings'][stage]:>8.1f}")
    print(f"  {'wall-clock':<12} {cold_wall:>8.1f} {warm_wall:>8.1f}")

    with open("response_pipeline.wav", "wb") as f:
        f.write(cold["answer_audio"])
    print("\nAudio saved to response_pipeline.wav")
