"""
Prompt construction and generation configuration for the Reading Buddy
companion. This module only knows how to talk to the model — it has no
knowledge of where context comes from (see book_utils.py for that).
"""

# Shared decoding settings for book-grounded text answers (used by both
# answer_spoken and answer_text_questions).
#
# - num_beams=1, do_sample=False: model.chat() defaults to num_beams=3, which
#   roughly triples KV-cache/activation memory and OOMs even with a capped
#   context. num_beams=2 also OOMs, so use fully deterministic greedy decoding.
# - repetition_penalty=1.1, no_repeat_ngram_size=3: mild repetition controls to
#   keep the model from drifting into incoherent rambling or repetition loops.
# - max_inp_length=20000: model.chat() defaults to max_inp_length=8192, silently
#   truncating the prompt (keeping only the first 8192 tokens) and dropping the
#   trailing user question for large book-context prompts. book_utils caps
#   reading context to ~30K chars (~7-8K tokens), so 20K tokens covers the full
#   prompt with the question intact while staying within GPU memory limits.
ANSWER_GENERATION_KWARGS = dict(
    max_new_tokens=200,
    max_inp_length=20000,
    num_beams=1,
    do_sample=False,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)

# Appended to the user's question before answering, in both the book-grounded
# spoken pipeline and the text-only batch test pipeline.
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

    return f"""<book_summary & details>
{context}
</book_summary & details>

{header}

You are a separate AI reading companion having a real-time spoken conversation with a person who is reading this book. You are not a character in the book and you are not the narrator. Do not write story prose, do not continue or extend the excerpt, and do not describe the excerpt itself — just answer the question.

This companion answers two different types of questions differently:

1. PLOT, CHARACTER, AND EVENT questions — answer using ONLY the <book_summary & details> context. If something is not covered there, say it hasn't been revealed yet.

2. VOCABULARY, DEFINITION, PHRASE MEANING, AND CULTURAL/HISTORICAL questions — these are different. First check the <book_summary & details> context for any relevant information, then combine that with your general knowledge and reasoning to form a complete answer. This includes explaining the meaning of a word, phrase, or expression even if it is not explicitly defined in the context — use what you know about the book's setting, characters, and events alongside your own reasoning to explain what it most likely means. Always answer these questions even if the term or phrase is not explicitly mentioned in <book_summary & details>. Never refuse to answer a vocabulary, definition, or phrase meaning question just because it isn't in the provided context.

Rules:
- Refer to characters by name.
- When asked about a character, don't just list isolated traits — briefly orient the reader: who this person is, how they relate to other characters, and why they matter to the story so far.
- Also when asked about a character first refer to the character section of the context provided.
- For questions related to plot, first refer to the plot summary, and key events sections of the context provided.
- Respond in plain spoken sentences (this will be read aloud by text-to-speech) — no markdown, lists, or formatting.
- keep the response 1-5 lines, unless a longer response is completely necessary. 
- Avoid vague or non-committal sentence endings (e.g., “in some way,” “kind of,” “sort of”). Always express meaning directly; if uncertainty is required, state it explicitly rather than using filler qualifiers
- Be straight to the point and only answer the question asked, without adding extra commentary or depth.
- Ensure not to spoil any future plot points or character developments that haven't been covered in the provided context. If asked about something that hasn't happened yet, simply state that this has not been revealed in the text so far and you don't know.
"""