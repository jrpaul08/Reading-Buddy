"""
One-time book preprocessing pipeline for Reading Buddy.

Uses MiniCPM4.1-8B (text-only) to generate per-chapter summaries and
structured data (characters, key events, locations, cultural references)
for each book, producing a summary_key_data.json matching the hand-written
reference format already used by book_utils.load_hybrid_context.

This is a standalone utility, separate from the production modal_inference.py
app and model.

Usage:
    modal run preprocess_books.py::test_model
"""

import modal

app = modal.App("reading-buddy-preprocess")

MODEL_ID = "openbmb/MiniCPM4.1-8B"
MODEL_DIR = "/model-weights"

vol = modal.Volume.from_name("reading-buddy-preprocess-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.56.0",
        "accelerate",
        "einops",
        "huggingface_hub",
        "sentencepiece",
    )
)


@app.cls(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: vol},
    scaledown_window=600,
    timeout=600,
)
class ChapterSummarizer:

    @modal.enter()
    def enter(self):
        """
        Purpose: Runs once when the Modal container starts. Downloads
        MiniCPM4.1-8B weights to this pipeline's dedicated Volume if not
        already present, then loads the model and tokenizer into GPU memory.

        Args:
            None

        Returns:
            None — sets self.model and self.tokenizer as instance attributes.
        """
        import os
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
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().cuda()
        print(f"[model load] {time.time() - t0:.1f}s")

        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
        )
        print(f"[tokenizer load] {time.time() - t1:.1f}s")

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Purpose: Generate a text response from a plain text prompt using
        MiniCPM4.1-8B's chat template.

        Args:
            prompt (str): The user message to send to the model.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The model's text response.
        """
        import time
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        t0 = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        print(f"[generate] {time.time() - t0:.1f}s")

        response_ids = output_ids[0][inputs.shape[1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)


@app.local_entrypoint()
def test_model():
    """
    Purpose: Validate that MiniCPM4.1-8B loads correctly on its dedicated
    volume/image and responds sensibly to a simple text prompt.

    Args:
        None

    Returns:
        None — result is printed to stdout.

    Usage:
        modal run preprocess_books.py::test_model
    """
    summarizer = ChapterSummarizer()
    response = summarizer.generate.remote("In one sentence, what is the capital of France?")
    print(f"\nResponse: {response}")
