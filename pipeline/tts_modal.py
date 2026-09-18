"""
Text-to-speech component of the Reading Buddy pipeline: Kokoro-82M,
running via the official `kokoro` PyPI package (KPipeline). Takes text,
returns spoken audio using a fixed voice ("af_bella") rather than
zero-shot cloning like the omni model — a deliberate trade accepted for
Kokoro's speed (see chat history for the CosyVoice2/Fish Speech
comparison this was weighed against).

Sources for design decisions in this file: see pipeline/SOURCES.md.

Built up incrementally, same shape as the other pipeline components:
    Part 1 (this file, current scope): prove the container starts,
        loads Kokoro onto GPU. No synthesis yet.
    Part 2: bare synthesis — text in, audio out, understand the
        (graphemes, phonemes, audio) generator shape KPipeline returns.
    Part 3: the full method the orchestrator will actually call.
"""

import os

import modal

from modal_app import app, vol

# All three pipeline components mount the shared volume at the same
# WEIGHTS_ROOT and separate their files via a subdirectory instead —
# see the comment in reasoning_modal.py for why a distinct mount path
# alone (what this used to do) doesn't actually separate anything.
WEIGHTS_ROOT = "/weights"
MODEL_DIR = os.path.join(WEIGHTS_ROOT, "kokoro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng")
    .pip_install(
        "torch",
        "kokoro>=0.9.4",
        "soundfile",
    )
    .env({"HF_HOME": MODEL_DIR})
    .add_local_file("pipeline/modal_app.py", "/root/modal_app.py")
)


@app.cls(
    gpu="L4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={WEIGHTS_ROOT: vol},
    scaledown_window=600,
    timeout=900,
)
class TTSEngine:

    @modal.enter()
    def enter(self):
        """
        Purpose: Runs once when the Modal container starts. Loads Kokoro
        onto the GPU via KPipeline, downloading its weights to the
        persistent Volume (via HF_HOME) on first run. All subsequent
        method calls on this container reuse the loaded pipeline.

        Args:
            None

        Returns:
            None — sets self.pipeline as an instance attribute.
        """
        import time
        from kokoro import KPipeline

        vol.reload()

        t0 = time.time()
        self.pipeline = KPipeline(lang_code="a", device="cuda")
        print(f"[pipeline load] {time.time() - t0:.1f}s")

    @modal.method()
    def ping(self) -> dict:
        """
        Purpose: Cheap sanity check that the pipeline actually loaded
        without error. KPipeline doesn't expose the same introspection
        PyTorch models do, so — same approach as STTEngine.ping — this
        reports the configuration requested rather than querying internal
        state. If this method returns at all, enter() succeeded.

        Args:
            None

        Returns:
            dict with keys:
                "pipeline_loaded" (bool): Always True if this returns.
                "device"           (str): The device requested at load
                    time.
                "lang_code"         (str): The language code requested,
                    "a" for American English.
        """
        return {
            "pipeline_loaded": True,
            "device": "cuda",
            "lang_code": "a",
        }

    @modal.method()
    def synthesize(self, text: str) -> bytes:
        """
        Purpose: Part 2 sanity check for real speech synthesis. Takes
        plain text and returns spoken audio as WAV bytes, proving the
        KPipeline synthesis chain works with the chosen voice
        ("af_bella") before Part 3 wires this into the shape the
        orchestrator will actually call.

        Args:
            text (str): The text to speak aloud.

        Returns:
            bytes: WAV-encoded audio of the spoken text.
        """
        generator = self.pipeline(text, voice="af_bella", speed=0.85)

        audio_chunks = [audio for _, _, audio in generator]

        import io
        import numpy as np
        import soundfile as sf

        full_audio = np.concatenate(audio_chunks)

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format="WAV")
        return buffer.getvalue()
