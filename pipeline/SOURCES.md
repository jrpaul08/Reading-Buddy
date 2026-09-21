# Model & library documentation

Core docs for the models/libraries each pipeline component actually
runs on, for maintenance reference.

## Reasoning — Qwen2.5-14B-Instruct

Standard `transformers` (`AutoModelForCausalLM`, `apply_chat_template`,
`generate`) — no non-default runtime.

- [Qwen2.5-14B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

## STT — Whisper large-v3-turbo via faster-whisper

Not run through `transformers`. Uses CTranslate2 (via the
`faster-whisper` package) instead — a separate, purpose-built inference
runtime, not the general-purpose library used for the reasoning
component.

- [faster-whisper GitHub (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper) — the runtime itself, `WhisperModel` API
- [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) — the CTranslate2-converted model actually loaded
- [faster-whisper issue #717](https://github.com/SYSTRAN/faster-whisper/issues/717) — `libcublas.so.12` fix (CTranslate2 doesn't bundle CUDA libs like torch does)

## TTS — Kokoro-82M via KPipeline

Not run through `transformers` either. Uses the official `kokoro`
package's `KPipeline` class — a Kokoro-specific entry point that bundles
model loading, text-to-phoneme conversion (G2P), and voice selection
into one object, and also handles its own model downloading internally
rather than an explicit `snapshot_download` call.

Calling a loaded `KPipeline` instance (`pipeline(text, voice=...)`)
returns a lazy generator, not a finished result — same shape as
`faster-whisper`'s `segments`, but yielding `(graphemes, phonemes,
audio)` per chunk instead of `(start, end, text)`. Kokoro splits input
text into pieces (roughly by sentence) and synthesizes each separately.

**Custom pronunciation overrides** — for words the default G2P
(text-to-phoneme) pipeline mispronounces, e.g. foreign proper nouns:
`pipeline.g2p.lexicon.golds['word'] = 'IPA_STRING'`, set on the loaded
pipeline object. The key is lowercase, the value is an IPA (International
Phonetic Alphabet) pronunciation string. Example from Kokoro's own demo
app: `pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'`.

The IPA string is not standard IPA — it uses Misaki's own phoneme set,
where diphthongs are single capital letters (`A`=eɪ, `I`=aɪ, `W`=aʊ,
`Y`=ɔɪ, `O`=oʊ in American English), and a few consonants are
respelled (`ɹ` for r, `ɡ` for g, `ʧ`=tʃ, `ʤ`=dʒ). Standard IPA copied
from a dictionary must be converted before use.

- [Misaki EN_PHONES.md](https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md) — full English phoneme inventory
- [hexgrad/misaki](https://github.com/hexgrad/misaki) — G2P library; also documents inline overrides `[word](/phonemes/)`

- [kokoro on PyPI](https://pypi.org/project/kokoro/) — install, dependencies, `KPipeline` usage, what it bundles (G2P + voice management + generation), the generator return shape
- [hexgrad/kokoro GitHub](https://github.com/hexgrad/kokoro) — source, voice list
- [hexgrad/Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)

## TODO

- **Character name pronunciation (TTS).** Kokoro mispronounces foreign
  character names (e.g. "Raskolnikov"). Deferred deliberately: the fix
  belongs in the book-preprocessing pipeline (`preprocess_books.py`),
  generating a pronunciation per character alongside the existing
  character data, not a hand-maintained table in `tts_modal.py`. The
  mechanism (`pipeline.g2p.lexicon.golds`) and the required phoneme
  notation are documented in the Kokoro section above.

### Orchestrator (`orchestrator_modal.py`)

- **Empty transcription.** A silent or empty recording makes STT return
  `""`, and the orchestrator currently sends that on to Qwen anyway.
  Decide what should happen instead (error, retry prompt, canned reply).
- **`chapter < 1`.** Produces an empty chapter list, and `book_utils`
  then fails with an `IndexError` rather than a clean `ValueError`. The
  omni endpoint had the same gap. Low priority unless the frontend can
  ever send it.
- **`ValueError` is ambiguous in `run_pipeline`.** It covers both an
  unknown `book_id` (a bad request, should be a 400) and TTS producing no
  audio (a server-side problem, should not be a 400). omni's endpoint
  turned every `ValueError` into a 400. The `s2s_endpoint` piece needs to
  tell them apart.
- **Cold vs warm timings.** Each stage's timing includes that
  container's cold start when it was cold, so a single run says little
  about steady-state speed. Compare cold and warm runs before drawing
  conclusions.

### Speed ideas (future)

- **Overlap the stages.** The pipeline is strictly sequential, so
  end-to-end latency is the sum of STT + reasoning + TTS: TTS can't start
  until Qwen has written the whole answer. The one large lever beyond
  warm containers would be streaming: start speaking the first sentence
  while Qwen is still writing the second. Substantial change (streaming
  generation, chunked audio back to the frontend), so not planned yet.

### Verification still owed

- **spaCy model baked into the TTS image.** Confirm that the
  `Collecting en-core-web-sm ... Downloading` block no longer appears
  in a TTS run's output. That would prove Kokoro finds the pre-installed
  model instead of downloading it on every cold start.
- **Orchestrator mounts.** The mount list in `orchestrator_modal.py` was
  derived by tracing imports, not tested. The first run confirms it.
