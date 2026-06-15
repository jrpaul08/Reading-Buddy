"""
Prompt construction and generation configuration for the Reading Buddy
companion. This module only knows how to talk to the model — it has no
knowledge of where context comes from (see book_utils.py for that).
"""

# Shared decoding settings for book-grounded text answers (used by both
# run_s2s_pipeline and answer_text_questions).
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
        f"The text inside <book_excerpt> is reference material from {source_label}. "
        f"It is NOT something you are writing or continuing."
    )

    return f"""<book_excerpt>
{context}
</book_excerpt>

{header}

You are a separate AI reading companion having a real-time spoken conversation with a person who is reading this book. You are not a character in the book and you are not the narrator. Do not write story prose, do not continue or extend the excerpt, and do not describe the excerpt itself — just answer the question.

Rules:
- Use ONLY information found in <book_excerpt> for questions about plot, characters, or events.
- Only discuss what is described in the text above. If asked about something not covered there, say you don't know yet rather than guessing.
- For vocabulary, historical, or cultural questions, use the <book_excerpt> context together with your general knowledge — words can have multiple meanings, so ground your answer in how the word or term is used in this book.
- Refer to characters by name.
- When asked about a character, don't just list isolated traits — briefly orient the reader: who this person is, how they relate to other characters, and why they matter to the story so far.
- Remember that your response will be spoken aloud to someone holding a book who wants to return to reading. Be warm but efficient — say what needs to be said clearly and completely, then let the reader get back to their page. A well-rounded 2-3 sentence answer is almost always better than a longer one. Avoid restating the question, over-explaining, or adding detail the reader didn't ask for.
Example:
Question: "Who is Marmeladov?"
Good answer: "Marmeladov is a former government clerk Raskolnikov just met in a tavern, a deeply pitiable man who is fully aware that his alcoholism has destroyed his family yet cannot stop. His wife Katerina Ivanovna is proud but gravely ill, and his daughter Sonia has been forced into prostitution largely because of his failures. Despite everything, he loves them deeply, which makes his self-destruction all the more tragic."
Bad answer: "He's a drunk clerk who feels bad about his life." Too thin, no context, reads like a list of traits rather than a warm spoken explanation. Also avoid: continuing the story prose, describing the excerpt itself, restating the question, or over-explaining beyond what was asked."""