"""
Text-to-speech component of the Reading Buddy pipeline: Kokoro-82M,
running via the official `kokoro` PyPI package (KPipeline). Takes text,
returns spoken audio using a fixed voice ("af_bella") rather than
zero-shot cloning like the omni model — a deliberate trade accepted for
Kokoro's speed (see chat history for the CosyVoice2/Fish Speech
comparison this was weighed against).

Sources for design decisions in this file: see pipeline/SOURCES.md.

Built up incrementally, same shape as the other pipeline components:
    Part 1: prove the container starts and loads Kokoro onto GPU.
    Part 2 (current scope): synthesis. synthesize(text) -> WAV bytes is
        already the shape an orchestrator needs to call, so there is no
        separate Part 3.

Voice is af_bella at speed 0.85, chosen by ear. Known gap: foreign
character names are mispronounced — deliberately deferred, see the TODO
in pipeline/SOURCES.md.
"""

import os

import modal

from modal_app import app, vol

MODEL_ID = "hexgrad/Kokoro-82M"

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
    # Kokoro's text processing needs this spaCy model. Left alone, it runs
    # `pip install` for it at runtime on every cold start (seen in the
    # first TTS run's logs). Installing it here bakes it into the image.
    # Same wheel, same version that the runtime download fetched.
    .pip_install(
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
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
        self.pipeline = KPipeline(lang_code="a", repo_id=MODEL_ID, device="cuda")
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
        Purpose: Speaks text aloud. Takes plain text and returns the
        spoken audio as WAV bytes (24 kHz), using the fixed voice
        ("af_bella") at speed 0.85. Kokoro splits long text into chunks
        and synthesizes each separately; the chunks are joined into one
        continuous clip here.

        Args:
            text (str): The text to speak aloud.

        Returns:
            bytes: WAV-encoded audio of the spoken text.

        Raises:
            ValueError: If Kokoro produces no audio for the input (empty,
                whitespace-only, or otherwise unspeakable text). The
                caller decides how to handle this.
        """
        generator = self.pipeline(text, voice="af_bella", speed=0.85)

        audio_chunks = [audio for _, _, audio in generator]

        if not audio_chunks:
            raise ValueError(
                f"Kokoro produced no audio for the given text: {text!r}"
            )

        import io
        import numpy as np
        import soundfile as sf

        full_audio = np.concatenate(audio_chunks)

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format="WAV")
        return buffer.getvalue()
