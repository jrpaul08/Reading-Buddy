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
