"""
Shared Modal App and Volume for the Reading Buddy three-model pipeline
(Whisper STT + Qwen reasoning + Kokoro TTS). Each component's module
(reasoning_modal.py, stt_modal.py, tts_modal.py, orchestrator_modal.py)
imports `app` from here so they all register on the same Modal App, while
still deploying and scaling as independent classes/containers.

Kept as "reading-buddy-pipeline" (not "reading-buddy") while this
implementation is under development, so it can be redeployed freely without
affecting the live omni model, which currently owns the "reading-buddy"
app name. Once this pipeline is validated end to end, redeploying it under
"reading-buddy" is what makes it the production backend (see omni/ for the
model currently live there).
"""

import modal

app = modal.App("reading-buddy-pipeline")

# Caches downloaded model weights (Whisper, Qwen, Kokoro) across container
# restarts, so a redeploy or cold start doesn't re-download gigabytes of
# weights every time. Kept separate from omni's "reading-buddy-weights"
# volume, which only holds MiniCPM-o's weights — different model files,
# no reason to share a volume between the two implementations.
vol = modal.Volume.from_name("reading-buddy-pipeline-weights", create_if_missing=True)
