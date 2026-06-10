# Reading Buddy — Project Context

> Read this file before writing any code or making any architectural decisions.

---

## What This Project Is

Reading Buddy is a voice-first AI reading companion. A reader sits with a physical book, speaks a question out loud, and gets an immediate spoken answer — without spoilers, without breaking their reading flow.

**Core philosophy: help the reader continue reading.**

---

## The Problem

Reading classic literature creates repeated friction:
- Unfamiliar vocabulary and sentence structures
- Large casts of characters with multiple names and aliases
- Historical and cultural references
- Confusing plot developments

Searching the internet breaks immersion and introduces distractions. Reading Buddy eliminates this through immediate, voice-based, spoiler-free assistance.

---

## Hackathon Constraints

| Field | Value |
|---|---|
| Event | Build Small Hackathon — HuggingFace + Gradio |
| Track | Backyard AI |
| Deadline | June 15, 2026 |
| Platform | Gradio app hosted on HuggingFace Space |
| Model limit | ≤ 32 billion parameters |
| Submission | HF Space link + demo video + social media post |

---

## Core Technical Concept — Spoiler Prevention

The most important feature of this product.

Each book is pre-split into chapters. When the user selects their current chapter N, only chapters 1 through N are injected into the model context. The model cannot spoil future events because it has not been given them. This is an architectural constraint, not a prompt instruction.

```python
def get_reading_context(book: str, chapter: int) -> str:
    chapters = load_book_chapters(book)
    return "\n\n".join(chapters[:chapter])
```

---

## Book Data

All books sourced from Project Gutenberg (public domain only).

Each book requires:
- `chapters.json` — full text split by chapter
- `characters.json` — character names, aliases, and the chapter they are introduced

The `introduced_chapter` field on each character must be respected — never surface a character before they appear in the story.

---

## Infrastructure

| Component | Technology |
|---|---|
| GPU inference | Modal (workspace: pauljared48) |
| App framework | Gradio |
| Hosting | HuggingFace Space (private during dev) |
| Book data | Project Gutenberg plain text |
| Version control | GitHub (primary), HF Space (synced via GitHub Action) |

---
## Environment
| Tool | Status |
|---|---|
| Python 3.11 (Homebrew) | ✅ |
| Virtual environment | ✅ |
| Gradio installed | ✅ |
| GitHub repo | ✅ |
| HF Space (private) | ✅ |
| GitHub → HF sync | ✅ |
| Modal installed + authenticated | ✅ |
---

## Key Links

| Resource | URL |
|---|---|
| HF Space | https://huggingface.co/spaces/jrpaul08/Reading-Buddy |
| Modal Dashboard | https://modal.com/apps/pauljared48 |
| Hackathon Page | https://huggingface.co/build-small-hackathon |
| Project Gutenberg | https://www.gutenberg.org |

---