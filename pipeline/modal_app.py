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

# Storage for the "save the response" feature: each saved item (question,
# answer, book_id, chapter, saved_at) lives in a list keyed by whichever
# id the caller resolved to — a verified Clerk user_id if logged in, or
# a guest-made-up session_id if not (see
# orchestrator_modal._resolve_identity) — not used anywhere else in the
# pipeline, which otherwise stays stateless.
saved_responses = modal.Dict.from_name("reading-buddy-pipeline-saved-responses", create_if_missing=True)

# Storage for the "add to glossary" feature: each entry (term,
# definition, book_id, chapter, saved_at) lives in a list keyed the same
# way as saved_responses. Only ever written to when
# ReasoningEngine.check_vocabulary confirms an exchange was actually a
# vocabulary question — never written to directly from user input.
glossary_entries = modal.Dict.from_name("reading-buddy-pipeline-glossary-entries", create_if_missing=True)

# Storage for reading position: keyed the same way as saved_responses/
# glossary_entries, but the value is a dict of {book_id: chapter} rather
# than a list — someone can be partway through more than one book at
# once, so position is tracked per book, not as a single value.
reading_progress = modal.Dict.from_name("reading-buddy-pipeline-reading-progress", create_if_missing=True)
