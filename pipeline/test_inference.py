"""
Debug/experimental entrypoints for the Reading Buddy pipeline's reasoning
component. Mirrors the structure of omni/test_inference.py, built up
incrementally alongside pipeline/reasoning_modal.py.

Usage:
    modal run pipeline/test_inference.py::check_load
    modal run pipeline/test_inference.py::ask_question
"""

from reasoning_modal import app, ReasoningEngine


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
