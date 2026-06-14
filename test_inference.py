"""
Debug/experimental entrypoints for Reading Buddy. Imports the shared Modal
app, image, and ReadingCompanion class from modal_inference.py so these
register on the same app, without cluttering the production module.

Usage:
    modal run test_inference.py::check_truncation
    modal run test_inference.py::check_prompt
    modal run test_inference.py::test_context_qa
    modal run test_inference.py::test_book
    modal run test_inference.py::test_book_hybrid
"""

import json
from pathlib import Path

from modal_inference import app, image, MODEL_DIR, vol, ReadingCompanion


@app.function(image=image, volumes={MODEL_DIR: vol})
def inspect_chat_truncation() -> str:
    """
    Purpose: Cheap CPU-only diagnostic. Searches the cached MiniCPM-o-4_5
    remote-code files on the model volume for any input-length truncation
    logic (e.g. max_inp_length) used inside model.chat(), to determine
    whether long prompts get silently truncated before generation.

    Args:
        None

    Returns:
        str: Matching lines (with filename and line number) from any .py
            file under MODEL_DIR whose content mentions length-related
            truncation keywords.
    """
    import os
    import re

    keywords = re.compile(r"max_inp_length|max_input|truncat|max_new_tokens\s*=|model_max_length|num_beams")
    matches = []

    for root, _, files in os.walk(MODEL_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if keywords.search(line):
                            matches.append(f"{fpath}:{i}: {line.rstrip()}")
            except OSError:
                continue

    return "\n".join(matches) if matches else "No matches found."


@app.function(image=image, volumes={MODEL_DIR: vol})
def check_prompt_size(book_name: str, chapter_number: int) -> dict:
    """
    Purpose: Cheap CPU-only diagnostic. Builds the book-grounded system prompt for
    the given book/chapter and tokenizes it to check how many tokens it uses
    relative to the model's max context length, without spinning up the GPU.

    Args:
        book_name (str): Book identifier (e.g. "crime_and_punishment").
        chapter_number (int): The reader's current chapter (1-indexed).

    Returns:
        dict with keys:
            "prompt_chars"     (int): Character length of the system prompt.
            "prompt_tokens"    (int): Token length of the system prompt.
            "model_max_length" (int): The tokenizer's reported max sequence length.
    """
    from transformers import AutoTokenizer
    from book_utils import describe_reading_context
    from prompt_utils import build_system_prompt

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    context, source_label = describe_reading_context(book_name, chapter_number)
    prompt = build_system_prompt(context, source_label)
    prompt_tokens = len(tokenizer.encode(prompt))

    return {
        "prompt_chars": len(prompt),
        "prompt_tokens": prompt_tokens,
        "model_max_length": tokenizer.model_max_length,
    }


@app.local_entrypoint()
def check_truncation():
    """
    Purpose: Local entrypoint for the inspect_chat_truncation diagnostic.
    Prints any truncation/max-length related lines found in the cached
    MiniCPM-o-4_5 remote-code files, with no GPU cost.

    Args:
        None

    Returns:
        None — results are printed to stdout.

    Usage:
        modal run test_inference.py::check_truncation
    """
    print(inspect_chat_truncation.remote())


@app.local_entrypoint()
def check_prompt():
    """
    Purpose: Local entrypoint for the check_prompt_size diagnostic. Reports the
    token count of the chapter 1-7 "crime_and_punishment" system prompt against
    the model's max context length, with no GPU cost.

    Args:
        None

    Returns:
        None — results are printed to stdout.

    Usage:
        modal run test_inference.py::check_prompt
    """
    result = check_prompt_size.remote("crime_and_punishment", 7)
    print(result)


@app.local_entrypoint()
def test_context_qa(save_results: bool = False):
    """
    Purpose: Local entrypoint to test the generated hybrid (summary +
    structured-data) context for chapters 1-2 of "Crime and Punishment"
    against a fixed list of questions, using the text-only inference pipeline
    (no audio, no TTS).
    Prints each question alongside its answer, and optionally writes the same
    results to a timestamped JSON file under test_results/ so runs can be
    compared later.

    Args:
        save_results (bool): If True, save results to test_results/. Defaults
            to False.

    Returns:
        None — results are printed to stdout, and saved to test_results/ if
        save_results is True.

    Usage:
        modal run test_inference.py::test_context_qa
        modal run test_inference.py::test_context_qa --save-results
    """
    from book_utils import describe_hybrid_context

    book_name = "crime_and_punishment"
    context, source_label = describe_hybrid_context(book_name, [1, 2])

    questions_path = Path(__file__).parent / "books" / book_name / "test_questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    companion = ReadingCompanion()
    results = companion.answer_text_questions.remote(
        context=context,
        questions=questions,
        source_label=source_label,
    )

    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")

    if save_results:
        import datetime

        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"context_qa_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(
                {"context": context, "results": results},
                f,
                indent=2,
            )
        print(f"\nSaved results to {output_path}")


@app.local_entrypoint()
def test_book(chapter_number: int = 7, audio_file: str = "voice-prompts/voice-prompt-ch7.wav"):
    """
    Purpose: Local test entrypoint for the book-grounded spoken response pipeline.
    Reads the given audio file, sends it to run_s2s_pipeline along with book context
    for "crime_and_punishment" up to the given chapter, prints the transcribed
    question and text answer, and saves the spoken audio response to
    response_book.wav.

    Args:
        chapter_number (int): The reader's current chapter (1-indexed). Defaults to 7.
        audio_file (str): Path to the input audio file. Defaults to
            "voice-prompts/voice-prompt-ch7.wav".

    Returns:
        None — results are printed to stdout and audio is saved to response_book.wav.

    Usage:
        modal run test_inference.py::test_book
        modal run test_inference.py::test_book --chapter-number 4
        modal run test_inference.py::test_book --chapter-number 2 --audio-file voice-prompts/voice-prompt-ch2.wav
    """
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    companion = ReadingCompanion()
    result = companion.run_s2s_pipeline.remote(audio_bytes, book_name="crime_and_punishment", chapter_number=chapter_number)

    print(f"\nQuestion heard: {result['question']}")
    print(f"\nAnswer: {result['answer_text']}")

    with open("response_book.wav", "wb") as f:
        f.write(result["answer_audio"])
    print("\nAudio response saved to response_book.wav")


@app.local_entrypoint()
def test_book_hybrid(chapter_number: int = 2, audio_file: str = "voice-prompts/voice-prompt-ch7.wav"):
    """
    Purpose: Local test entrypoint for the book-grounded spoken response pipeline
    using the hybrid (summary + structured data) context from
    book_chapter_context.json, instead of raw chapter text. Useful for checking
    spoiler prevention: e.g. asking a question whose answer is revealed in a later
    chapter, while only providing context for earlier chapters.

    Args:
        chapter_number (int): The reader's current chapter (1-indexed). Context
            includes chapters 1 through this chapter, inclusive. Defaults to 2.
        audio_file (str): Path to the input audio file. Defaults to
            "voice-prompts/voice-prompt-ch7.wav".

    Returns:
        None — results are printed to stdout and audio is saved to response_book.wav.

    Usage:
        modal run test_inference.py::test_book_hybrid
        modal run test_inference.py::test_book_hybrid --chapter-number 7 --audio-file voice-prompts/voice-prompt-ch7.wav
    """
    chapters = list(range(1, chapter_number + 1))

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    companion = ReadingCompanion()
    result = companion.run_s2s_pipeline.remote(audio_bytes, book_name="crime_and_punishment", chapter_numbers=chapters)

    print(f"\nQuestion heard: {result['question']}")
    print(f"\nAnswer: {result['answer_text']}")

    with open("response_book.wav", "wb") as f:
        f.write(result["answer_audio"])
    print("\nAudio response saved to response_book.wav")
