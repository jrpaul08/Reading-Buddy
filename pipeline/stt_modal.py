"""
Speech-to-text component of the Reading Buddy pipeline: Whisper large-v3
turbo, running via faster-whisper (a CTranslate2-based runtime built
specifically for Whisper, not the general-purpose transformers library
used for the reasoning component). Takes raw audio bytes, returns a
transcription.

Built up incrementally, same shape as reasoning_modal.py:
    Part 1 (this file, current scope): prove the container starts,
        downloads the model, and loads it onto GPU. No transcription yet.
    Part 2: bare transcription — feed it real audio, get text back.
    Part 3: real-world audio handling (WebM-from-browser conversion,
        resampling), ported from omni's _transcribe.
    Part 4: the full method the orchestrator will actually call.
"""

import os

import modal

from modal_app import app, vol

MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Own directory on the shared pipeline volume — kept distinct from Qwen's
# "/model-weights" so two unrelated models' files never land in the same
# directory, even though both live on the same underlying Volume.
MODEL_DIR = "/stt-weights"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Needed in Part 3 for converting browser-recorded WebM audio to a
    # format faster-whisper can read — added now so there's no image
    # rebuild needed when we get there.
    .apt_install("ffmpeg")
    .pip_install(
        # Note: no torch here. faster-whisper runs on CTranslate2, a
        # separate C++/CUDA inference engine — it doesn't depend on
        # PyTorch at all, unlike the reasoning component.
        "faster-whisper",
        "huggingface_hub",
        # Unlike torch, CTranslate2 doesn't bundle its own CUDA runtime
        # libraries — it expects libcublas/libcudnn to already be on the
        # system. These packages just ship those shared library files.
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
    })
    .add_local_file("pipeline/modal_app.py", "/root/modal_app.py")
)


@app.cls(
    gpu="L4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: vol},
    scaledown_window=600,
    timeout=900,
)
class STTEngine:

    @modal.enter()
    def enter(self):
        """
        Purpose: Runs once when the Modal container starts. Downloads
        Whisper large-v3-turbo (CTranslate2 format) to the persistent
        Volume if not already present, then loads it onto the GPU. All
        subsequent method calls on this container reuse the loaded model.

        Args:
            None

        Returns:
            None — sets self.model as an instance attribute.
        """
        from huggingface_hub import snapshot_download

        vol.reload()

        if not os.path.exists(os.path.join(MODEL_DIR, "model.bin")):
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_DIR,
                token=os.environ["HF_TOKEN"],
            )
            vol.commit()

        import time
        from faster_whisper import WhisperModel

        t0 = time.time()
        self.model = WhisperModel(MODEL_DIR, device="cuda", compute_type="float16")
        print(f"[model load] {time.time() - t0:.1f}s")

    @modal.method()
    def ping(self) -> dict:
        """
        Purpose: Cheap sanity check that the model actually loaded without
        error. faster-whisper's WhisperModel doesn't expose the same
        introspection PyTorch models do (no .parameters() to inspect), so
        this reports the configuration we requested rather than querying
        internal state — if this method returns at all, enter() succeeded.

        Args:
            None

        Returns:
            dict with keys:
                "model_loaded" (bool): Always True if this method returns.
                "device"        (str): The device requested at load time.
                "compute_type"  (str): The precision requested at load
                    time, e.g. "float16".
        """
        return {
            "model_loaded": True,
            "device": "cuda",
            "compute_type": "float16",
        }

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Purpose: Part 2 sanity check for real transcription. Takes raw
        audio bytes and returns the transcribed text, proving the
        faster-whisper transcription chain works before Part 3 adds
        real-world audio format handling (WebM conversion, resampling).

        Args:
            audio_bytes (bytes): Raw audio file content (WAV for now —
                Part 3 will add support for other formats).

        Returns:
            str: The transcribed text.
        """
        import io

        audio_file = io.BytesIO(audio_bytes)

        segments, info = self.model.transcribe(audio_file)

        transcription = " ".join(segment.text.strip() for segment in segments)
        return transcription
