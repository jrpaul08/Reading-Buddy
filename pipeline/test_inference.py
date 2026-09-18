"""
Debug/experimental entrypoints for the Reading Buddy pipeline. Mirrors the
structure of omni/test_inference.py, built up incrementally alongside
each pipeline component (reasoning_modal.py, stt_modal.py, ...).

Usage:
    modal run pipeline/test_inference.py::check_load
    modal run pipeline/test_inference.py::ask_question
    modal run pipeline/test_inference.py::test_book
    modal run pipeline/test_inference.py::check_stt_load
"""

import json
from pathlib import Path

from reasoning_modal import app, ReasoningEngine
from stt_modal import STTEngine


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
        questions (str): Questions separated by | (pipe). If omitted,
            falls back to the full list in
            books/<book_name>/test_questions.json — the same file
            omni/test_inference.py's test_context_qa reads, so editing
            that one list improves test coverage for both models at once.

    Returns:
        None — results are printed to stdout.

    Usage:
        modal run pipeline/test_inference.py::test_book
        modal run pipeline/test_inference.py::test_book --chapter-number 7 --questions "Who is Sonya?|What is the axe for?"
    """
    from book_utils import describe_hybrid_context

    chapter_numbers = list(range(1, chapter_number + 1))
    context, source_label = describe_hybrid_context(book_name, chapter_numbers)

    if questions:
        question_list = [q.strip() for q in questions.split("|") if q.strip()]
    else:
        questions_path = Path(__file__).parent.parent / "books" / book_name / "test_questions.json"
        with open(questions_path) as f:
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
