"""
Reasoning component of the Reading Buddy pipeline: Qwen2.5-14B-Instruct,
running as its own Modal class. Takes book context + a question, returns a
spoiler-free text answer. Counterpart to omni/modal_inference.py's
ReadingCompanion, but specialized to reasoning only (no STT/TTS).

Built up incrementally:
    Part 1 (this file, current scope): prove the container starts,
        downloads the model, and loads it onto GPU. No generation yet.
    Part 2: bare text generation, no book context/system prompt.
    Part 3: real generation settings (pipeline/prompt_utils.py).
    Part 4: the full answer(context, questions, source_label) method.
"""

import os

import modal

from modal_app import app, vol

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_DIR = "/model-weights"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Pinned to the same version already proven working in this
        # environment for the omni model, to keep one less variable in
        # play while validating a new model/GPU combination.
        "transformers==4.51.0",
        "accelerate",
        "huggingface_hub",
        "safetensors",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.cls(
    gpu="L40S",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: vol},
    scaledown_window=600,
    timeout=900,
)
class ReasoningEngine:

    @modal.enter()
    def enter(self):
        """
        Purpose: Runs once when the Modal container starts. Downloads
        Qwen2.5-14B-Instruct weights to the persistent Volume if not
        already present, then loads the model and tokenizer onto the GPU.
        All subsequent method calls on this container reuse the already-
        loaded model.

        Args:
            None

        Returns:
            None — sets self.model and self.tokenizer as instance
            attributes.
        """
        import time
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        vol.reload()

        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_DIR,
                token=os.environ["HF_TOKEN"],
            )
            vol.commit()

        t0 = time.time()
        # No BitsAndBytes quantization here, so .cuda() must be called
        # explicitly — from_pretrained() alone leaves the model on CPU,
        # running at CPU speed with no error raised to warn you.
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().cuda()
        print(f"[model load] {time.time() - t0:.1f}s")

        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print(f"[tokenizer load] {time.time() - t1:.1f}s")

    @modal.method()
    def ping(self) -> dict:
        """
        Purpose: Cheap sanity check that the model actually loaded and
        landed on the GPU (not silently stuck on CPU). No generation,
        just inspects the loaded model's device placement and size.

        Args:
            None

        Returns:
            dict with keys:
                "model_loaded" (bool): Always True if this method returns
                    at all (enter() would have raised otherwise).
                "device"        (str): The device the model's parameters
                    are on, e.g. "cuda:0". Should never be "cpu".
                "dtype"         (str): The model's parameter dtype, e.g.
                    "torch.bfloat16".
                "num_params"    (int): Total parameter count, as a sanity
                    check that the full 14B-parameter model loaded.
        """
        device = str(next(self.model.parameters()).device)
        dtype = str(next(self.model.parameters()).dtype)
        num_params = sum(p.numel() for p in self.model.parameters())

        return {
            "model_loaded": True,
            "device": device,
            "dtype": dtype,
            "num_params": num_params,
        }
