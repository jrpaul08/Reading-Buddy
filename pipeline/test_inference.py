"""
Debug/experimental entrypoints for the Reading Buddy pipeline's reasoning
component. Mirrors the structure of omni/test_inference.py, built up
incrementally alongside pipeline/reasoning_modal.py.

Usage:
    modal run pipeline/test_inference.py::check_load
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
