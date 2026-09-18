"""
Prompt construction and generation configuration for the Reading Buddy
pipeline's reasoning component (Qwen2.5-14B-Instruct). Counterpart to
omni/prompt_utils.py — the spoiler-prevention prompt rules are ported over
close to unchanged (they describe desired behavior, not anything specific
to MiniCPM), but the generation settings are chosen fresh for this model
rather than copied, since the old ones were tuned around MiniCPM-specific
bugs that don't apply here.

The actual prompt wording lives in system_prompt_template.txt, not inline
here — a plain-text file is much easier to read and edit than a Python
f-string, and keeps prompt-content changes as clean text diffs instead of
Python-syntax diffs.
"""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "system_prompt_template.txt"

ANSWER_GENERATION_KWARGS = dict(
    max_new_tokens=200,
    do_sample=False,
)

# Appended to the user's question before answering.
ANSWER_INSTRUCTION_SUFFIX = (
    "\n\n(Answer the question above directly, in your own words, with "
    "enough context for the reader to understand it. Do not "
    "continue or describe the book excerpt.)"
)


def build_system_prompt(context: str, source_label: str) -> str:
    """
    Build a complete system prompt for the AI reading companion from a
    context string and a description of where it came from.

    The context is wrapped in <book_excerpt> tags with strong "do not
    continue this story" framing placed immediately after it (closest to
    where generation begins). This is meant to break the model's tendency to
    treat the whole prompt as one continuous narrative to extend, which it
    does even when instructions are placed before the text.

    Args:
        context (str): The reference material to ground answers in — e.g.
            raw chapter text or a hand-written summary/structured-data block.
        source_label (str): Short description of what the context is and
            where it's from (e.g. 'the novel "Crime and Punishment" by
            Fyodor Dostoevsky (up to chapter 7 of 39)', or 'a summary of
            chapters 1-2 of "Crime and Punishment" by Fyodor Dostoevsky').

    Returns:
        str: Complete system prompt ready to be sent to the AI model.
    """
    header = (
        f"The text inside <book summary & details> is reference material from {source_label}. "
        f"It is NOT something you are writing or continuing."
    )

    template = TEMPLATE_PATH.read_text()
    return template.format(context=context, header=header)
