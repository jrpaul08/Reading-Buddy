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
from prompt_utils import ANSWER_GENERATION_KWARGS, ANSWER_INSTRUCTION_SUFFIX, build_system_prompt

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
    .add_local_file("pipeline/modal_app.py", "/root/modal_app.py")
    .add_local_file("pipeline/prompt_utils.py", "/root/prompt_utils.py")
    .add_local_file("pipeline/system_prompt_template.txt", "/root/system_prompt_template.txt")
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

    @modal.method()
    def ask(self, question: str) -> str:
        """
        Purpose: Part 2 sanity check for raw text generation. Sends a plain
        question straight to Qwen with no system prompt, no book context —
        just proving the tokenize -> generate -> decode chain works and
        produces a coherent answer, before Part 3/4 add real prompt
        structure and book-grounded context on top.

        Args:
            question (str): A plain question, e.g. "What is the capital
                of France?".

        Returns:
            str: The model's answer.
        """
        messages = [{"role": "user", "content": question}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer

    @modal.method()
    def answer(self, context: str, questions: list, source_label: str) -> list:
        """
        Purpose: Part 4 — the real book-grounded answer method. Given a
        context block (book text or hybrid summary), a list of questions,
        and a source label, builds the spoiler-prevention system prompt
        and answers each question using it, matching the same
        context/questions/source_label shape as omni's
        answer_text_questions so the two can be compared head to head.

        Args:
            context (str): Book text or summary to ground answers in.
            questions (list[str]): Questions to ask, each independent.
            source_label (str): Description of the context's source.

        Returns:
            list[dict]: One dict per question, each with keys "question"
            and "answer".
        """
        system_prompt = build_system_prompt(context, source_label)

        results = []
        for question in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{question}{ANSWER_INSTRUCTION_SUFFIX}"},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")

            import torch

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **ANSWER_GENERATION_KWARGS,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            results.append({"question": question, "answer": answer})

        return results
