"""
One-time book preprocessing pipeline for Reading Buddy.

Uses an LLM (text-only) to generate per-chapter summaries and
structured data (characters, key events, locations, cultural references)
for each book, producing a book_chapter_context.json in the format
consumed by book_utils.load_hybrid_context.

This is a standalone utility, separate from the production modal_inference.py
app and model.

Usage:
    modal run preprocess_books.py::test_model
"""

import json
import re
import unicodedata
from pathlib import Path

import modal

app = modal.App("reading-buddy-preprocess")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_DIR = "/model-weights"
MODEL_WEIGHTS_DIR = "/model-weights/qwen2.5-14b-instruct"

vol = modal.Volume.from_name("reading-buddy-preprocess-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.56.0",
        "accelerate",
        "einops",
        "huggingface_hub",
        "sentencepiece",
    )
)


def build_summarization_prompt(chapter_text: str, chapter_number: int) -> str:
    """
    Build a prompt instructing the model to summarize a single chapter into
    the structured JSON format used by book_chapter_context.json.

    Args:
        chapter_text (str): Full text of the chapter to summarize.
        chapter_number (int): 1-indexed chapter number being summarized.

    Returns:
        str: Complete prompt to send to the model.
    """
    return f"""Before generating your response, mentally walk through the chapter from beginning to end and identify every distinct scene. Your summary and key_events must cover all of them with equal attention, including the final scene.

You are processing chapter {chapter_number} of a novel for a reading companion app.
Your output will be used to answer readers' questions about the story so far — accuracy and detail matter.

Read the chapter text below and return a single JSON object with this exact structure:

{{
  "summary": "An in-depth summary that reads as a chronological walkthrough of the ENTIRE chapter, from its opening scene to its final scene — do not stop short. Do not miss any important details. Keep in mind the reader might have questions regarding any part of the story, so do not skip over any scene or interaction. Include specific character names, capture important internal reasoning or emotional moments, and explicitly describe how the chapter ends. Do not be vague — be specific about what happened and why. Make sure to include every single character that is presented or mentioned in the chapter regardless of how brief their appearance. Capture every action that a character does, whether to another character or as an individual action. Once written, re-read it and see what else can be included to add more depth.",

  "characters": [
    {{
      "name": "Character's most commonly used name",
      "aliases": ["Every other name, nickname, title, or patronymic used for this character anywhere in the chapter text"],
      "description": "2-3 sentences. Who this character is, their relationship to other characters, their personality or situation, and why they matter. Written for someone who has only read up to this chapter."
    }}
  ],

  "key_events": ["One specific action or moment per entry. Be granular — each entry should describe a single distinct thing that happened, not a summary of a whole scene. Name the specific characters involved. For a chapter with multiple scenes, aim for 8-12 events to ensure full coverage."],

  "locations": ["Distinct locations that appear in this chapter."],

  "cultural_references": [
    {{
      "term": "Any word, title, rank, institution, currency, disease, law, social custom, religious practice, period-specific object, or historical reference a modern reader may not understand",
      "explanation": "A two-part explanation: first, what this term means in its historical or cultural context using general knowledge — its actual significance, weight, or meaning in the world the novel is set in; second, how it specifically applies to the characters or situation in this chapter."
    }}
  ]
}}

After completing your summary, re-read it and ask yourself: have I described every specific action that occurs in the final scene with the same level of detail as the opening scene? If the answer is no, expand it. Vague closing phrases like "the scene concludes with" or "chaos ensued" are not acceptable — describe exactly what each character does, says, thinks, or feels at the end of the chapter.

Here is an example of the level of detail expected for "summary" and "key_events", for a chapter from a different novel (Jane Eyre):

"summary": "On a cold, rainy winter afternoon at Gateshead Hall, ten-year-old Jane Eyre has been excluded from the warm drawing-room by her aunt Mrs. Reed, who tells her she may not join the family until she becomes more sociable and pleasant. Jane slips into the adjoining breakfast-room alone, takes down Bewick's History of British Birds from the bookshelf, and retreats into the window-seat behind a thick red curtain, wrapping herself in quiet solitude. She reads contentedly, studying the dark wintry landscape through the glass and losing herself in the book's mysterious vignettes of desolate arctic coastlines, shipwrecks, and ghostly figures — images that match her own sense of cold isolation. Her peace is broken when John Reed, her fourteen-year-old cousin, enters the room calling for her. He is large, bullying, and physically repellent — a boy who torments Jane habitually while his mother turns a blind eye. Eliza, his sister, immediately betrays Jane's hiding place by pointing out the window-seat. Jane comes out trembling, knowing what is coming. John seats himself in an armchair and orders Jane to stand before him, then strikes her suddenly across the face without warning, telling her it is punishment for answering back to his mother and for her impudence. Jane endures the blow but when John picks up the heavy book she had been reading and hurls it at her, hitting her head and drawing blood, something breaks inside her. The pain overwhelms her fear and she flies at John in a frantic rage, calling him a murderer, a slave-driver, and a Roman emperor — comparisons she had privately drawn before but never dared speak aloud. John grabs her hair and shoulder and they grapple violently. Eliza and Georgiana run to fetch Mrs. Reed, who arrives with Bessie the nurse and the maid Abbot. The adults separate them, and all blame falls on Jane — she is described as a fury and a picture of passion while John's cruelty goes unacknowledged. Mrs. Reed orders Jane taken to the red-room and locked inside as punishment, and four hands immediately seize Jane and carry her upstairs."

"key_events": [
  "Mrs. Reed excludes Jane from the drawing-room and tells her she may not rejoin the family until she becomes more sociable and pleasant",
  "Jane slips into the breakfast-room alone and takes Bewick's History of British Birds from the bookshelf",
  "Jane hides behind the red curtain in the window-seat, reading contentedly and studying the dark winter landscape",
  "John Reed enters the breakfast-room calling for Jane, failing to find her himself",
  "Eliza betrays Jane's hiding place by pointing out the window-seat to John",
  "Jane comes out trembling and stands before John at his command",
  "John strikes Jane suddenly and hard across the face without warning",
  "John accuses Jane of having no right to the family's books and orders her to stand by the door",
  "John hurls the heavy book at Jane, hitting her and cutting her head against the door",
  "Jane calls John a murderer, a slave-driver, and a Roman emperor — comparisons she had privately drawn before but never spoken aloud",
  "John rushes at Jane grabbing her hair and shoulder, and Jane fights back frantically",
  "Eliza and Georgiana run to fetch Mrs. Reed",
  "Mrs. Reed arrives with Bessie and Abbot and the children are separated",
  "All blame is placed on Jane while John's cruelty goes unaddressed",
  "Mrs. Reed orders Jane taken to the red-room and locked inside as punishment",
  "Bessie and Abbot seize Jane with four hands and carry her upstairs"
]

Match this level of specificity and detail in your own summary and key_events, applied to the chapter below.

Important rules:
- Only extract information that is explicitly present in the chapter text. Do not use outside knowledge of the book.
- For aliases, scan every sentence for alternative names — patronymics, nicknames, titles, and shortened names all count.
- Every key event should name the specific characters involved.
- Include characters who are discussed or referenced in the chapter even if they do not physically appear. If a character's story, actions, or situation is described in detail by another character, they must be included in the characters list and their role must be reflected in the summary and key_events.
- This chapter may contain multiple distinct scenes or locations. Before writing the summary, mentally divide the chapter into its major scenes from beginning to end, and make sure the summary and key_events cover EVERY one of them, including the final scene — do not stop early or let one scene crowd out the rest of the chapter.
- For scenes involving confrontation, conflict, or high emotion, break them into individual moments in both the summary and key_events. Do not summarize a dramatic scene in one sentence — describe each distinct action that occurs within it.
- Preserve concrete details exactly as stated in the text — specific numbers, amounts, dates, durations, and names of people and places. Do not flatten these into vague generalizations.
- Cultural references are the one area where you may and should draw on general knowledge beyond the chapter text. For any term a modern reader might not understand — whether it is a social custom, legal concept, historical institution, disease, currency, religious practice, or period-specific object — provide an explanation grounded in the historical and cultural reality of the world the novel is set in, then connect it specifically to how it appears or affects characters in this chapter. Do not infer meaning or significance from the word alone — explain what it actually meant for people who lived in that world.
- Return ONLY the JSON object. No preamble, explanation, or markdown code fences.

<chapter_text>
{chapter_text}
</chapter_text>"""


def parse_chapter_json(raw_response: str) -> dict:
    """
    Parse and validate a model response as chapter summary JSON matching the
    book_chapter_context.json schema.

    Args:
        raw_response (str): Raw text returned by the model, which may
            include a <think>...</think> block and/or markdown code fences
            around the JSON.

    Returns:
        dict: Parsed chapter data with keys "summary", "characters",
            "key_events", "locations", "cultural_references".

    Raises:
        ValueError: If the response is not valid JSON, or doesn't match the
            expected schema.
    """
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    if not isinstance(data.get("summary"), str):
        raise ValueError("Missing or invalid 'summary' (expected string)")

    if not isinstance(data.get("characters"), list):
        raise ValueError("Missing or invalid 'characters' (expected list)")
    for c in data["characters"]:
        if not isinstance(c, dict):
            raise ValueError(f"Each character must be an object, got {type(c).__name__}")
        if not isinstance(c.get("name"), str):
            raise ValueError(f"Character missing 'name': {c}")
        if not isinstance(c.get("aliases"), list):
            raise ValueError(f"Character '{c.get('name')}' missing or invalid 'aliases' (expected list)")
        if not isinstance(c.get("description"), str):
            raise ValueError(f"Character '{c.get('name')}' missing or invalid 'description' (expected string)")

    if not isinstance(data.get("key_events"), list):
        raise ValueError("Missing or invalid 'key_events' (expected list)")

    if not isinstance(data.get("locations"), list):
        raise ValueError("Missing or invalid 'locations' (expected list)")

    if not isinstance(data.get("cultural_references"), list):
        raise ValueError("Missing or invalid 'cultural_references' (expected list)")
    for cr in data["cultural_references"]:
        if not isinstance(cr, dict):
            raise ValueError(f"Each cultural reference must be an object, got {type(cr).__name__}")
        if not isinstance(cr.get("term"), str):
            raise ValueError(f"Cultural reference missing 'term': {cr}")
        if not isinstance(cr.get("explanation"), str):
            raise ValueError(f"Cultural reference '{cr.get('term')}' missing or invalid 'explanation' (expected string)")

    return data


def _collect_character_names(data: list) -> dict:
    """
    Collect every unique character "name" appearing across all chapters,
    along with the union of all "aliases" ever recorded for that name.

    Args:
        data (list[dict]): Per-chapter entries, each with a "characters" list.

    Returns:
        dict: Maps each unique character name to a sorted list of aliases
            collected for that name across all chapters.
    """
    names = {}
    for entry in data:
        for c in entry["characters"]:
            names.setdefault(c["name"], set()).update(c.get("aliases", []))
    return {name: sorted(aliases) for name, aliases in names.items()}


def build_canonicalization_prompt(character_map: dict) -> str:
    """
    Build a prompt asking the model to group character name variants (as
    extracted independently per-chapter) by the actual person they refer to,
    and assign one canonical name per person.

    Args:
        character_map (dict): Maps each unique character name to its
            aliases, as returned by _collect_character_names.

    Returns:
        str: Complete prompt to send to the model.
    """
    entries = [{"name": name, "aliases": aliases} for name, aliases in character_map.items()]
    entries_json = json.dumps(entries, indent=2)

    return f"""The following is a list of characters extracted independently from each chapter of a novel, along with any aliases (nicknames, patronymics, titles, or alternate names) noted for them in that chapter. Because each chapter was processed separately, the SAME character may appear multiple times under different name variants (e.g. "Raskolnikov", "Rodion Romanovitch Raskolnikov", and "Rodya" might all refer to the same person).

Your task: group these entries by the actual person they refer to, and choose ONE canonical name for each person — use their most complete and recognizable form (e.g. full name with surname, or the name used most often).

Return a single JSON object mapping each "name" value below to its canonical name. Every name in the list below must appear as a key in your output. Two genuinely different characters must map to two different canonical names — do not merge distinct people just because they share a surname, title, or role.

Characters:
{entries_json}

Return ONLY the JSON object mapping each input name to its canonical name. No preamble, explanation, or markdown code fences."""


def parse_canonicalization_json(raw_response: str, expected_names: set) -> dict:
    """
    Parse and validate a model response as a character-name canonicalization
    mapping.

    Args:
        raw_response (str): Raw text returned by the model, which may
            include a <think>...</think> block and/or markdown code fences
            around the JSON.
        expected_names (set[str]): The set of character names that must
            appear as keys in the returned mapping.

    Returns:
        dict: Maps each input character name to its canonical name.

    Raises:
        ValueError: If the response is not valid JSON or doesn't match the
            expected schema. Names omitted from the mapping are not an
            error — they simply keep their original name (see
            apply_canonicalization's use of dict.get with a default).
    """
    text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        mapping = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e}")

    if not isinstance(mapping, dict):
        raise ValueError(f"Expected a JSON object, got {type(mapping).__name__}")

    for name, canonical in mapping.items():
        if not isinstance(canonical, str):
            raise ValueError(f"Mapping for '{name}' is not a string: {canonical!r}")

    missing = expected_names - set(mapping.keys())
    if missing:
        print(f"[parse_canonicalization_json] model omitted {len(missing)} name(s) from its mapping "
              f"(these will keep their original name): {sorted(missing)}")

    return mapping


def apply_canonicalization(data: list, mapping: dict) -> list:
    """
    Rewrite each chapter's "characters" list in place, replacing each
    character's "name" with its canonical name (per `mapping`) and merging
    "aliases" for characters that collapse onto the same canonical name
    within a chapter.

    Args:
        data (list[dict]): Per-chapter entries, each with a "characters" list.
        mapping (dict): Maps original character names to canonical names, as
            returned by parse_canonicalization_json.

    Returns:
        list[dict]: The same `data` list, with "characters" rewritten in place.
    """
    for entry in data:
        merged = {}
        order = []
        for c in entry["characters"]:
            canonical = mapping.get(c["name"], c["name"])
            aliases = set(c.get("aliases", []))
            if c["name"] != canonical:
                aliases.add(c["name"])
            aliases.discard(canonical)

            if canonical in merged:
                merged[canonical]["aliases"] = sorted(set(merged[canonical]["aliases"]) | aliases)
            else:
                merged[canonical] = {"name": canonical, "aliases": sorted(aliases), "description": c["description"]}
                order.append(canonical)

        entry["characters"] = [merged[name] for name in order]

    return data


def _normalize_name(name: str) -> str:
    """Strip diacritics and casing from a character name for fuzzy comparison
    (e.g. "Svidrigailov" and "Svidrigaïlov" both normalize to "svidrigailov")."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def merge_near_duplicate_names(data: list) -> list:
    """
    Catch character names that the canonicalization pass left as near-duplicates
    differing only by diacritics (e.g. the model dropped "ï" from "Svidrigaïlov"
    when generating its mapping, leaving "Svidrigailov" and "Svidrigaïlov" as
    separate canonical names). Groups current names by their diacritic-stripped
    form and merges each group onto the name that best preserves accents.

    Args:
        data (list[dict]): Per-chapter entries, each with a "characters" list,
            after an initial apply_canonicalization pass.

    Returns:
        list[dict]: The same `data` list, with any near-duplicate names merged.
    """
    groups = {}
    for entry in data:
        for c in entry["characters"]:
            groups.setdefault(_normalize_name(c["name"]), set()).add(c["name"])

    rename = {}
    for names in groups.values():
        if len(names) <= 1:
            continue
        canonical = max(names, key=lambda n: (any(ord(ch) > 127 for ch in n), len(n)))
        for name in names:
            if name != canonical:
                rename[name] = canonical

    if rename:
        print(f"Merging {len(rename)} near-duplicate name(s) left after canonicalization: {rename}")
        data = apply_canonicalization(data, rename)

    return data


def accumulate_aliases(data: list) -> list:
    """
    Make each character's "aliases" list cumulative across chapters: by
    chapter N, a character's aliases include every alias ever recorded for
    them in chapters 1..N (not just chapter N's own aliases). This lets
    load_hybrid_context's "last chapter wins" dedup show the full set of
    names a character has been called by up to that point.

    Args:
        data (list[dict]): Per-chapter entries, in chapter-number order,
            each with a "characters" list of canonicalized characters.

    Returns:
        list[dict]: The same `data` list, with "aliases" rewritten in place.
    """
    seen_aliases = {}
    for entry in sorted(data, key=lambda e: e["number"]):
        for c in entry["characters"]:
            running = seen_aliases.setdefault(c["name"], set())
            running.update(c.get("aliases", []))
            c["aliases"] = sorted(running)
    return data


def build_character_glossary(data: list) -> list:
    """
    Build a book-wide character glossary from canonicalized per-chapter data,
    with one entry per unique character including the earliest chapter they
    appear in.

    Args:
        data (list[dict]): Per-chapter entries (after canonicalization), each
            with "number" and a "characters" list.

    Returns:
        list[dict]: One entry per character, each with "name", "aliases",
            "description" (from the chapter where they first appear), and
            "introduced_chapter", sorted by introduced_chapter.
    """
    glossary = {}
    for entry in data:
        for c in entry["characters"]:
            name = c["name"]
            if name not in glossary:
                glossary[name] = {
                    "name": name,
                    "aliases": set(c["aliases"]),
                    "description": c["description"],
                    "introduced_chapter": entry["number"],
                }
            else:
                glossary[name]["aliases"].update(c["aliases"])
                glossary[name]["introduced_chapter"] = min(glossary[name]["introduced_chapter"], entry["number"])

    return [
        {
            "name": g["name"],
            "aliases": sorted(g["aliases"]),
            "description": g["description"],
            "introduced_chapter": g["introduced_chapter"],
        }
        for g in sorted(glossary.values(), key=lambda g: g["introduced_chapter"])
    ]


@app.cls(
    gpu="A100-40GB",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: vol},
    scaledown_window=600,
    timeout=600,
    max_containers=5,
)
class ChapterSummarizer:

    @modal.enter()
    def enter(self):
        """
        Purpose: Runs once when the Modal container starts. Downloads
        the model weights to this pipeline's dedicated Volume if not
        already present, then loads the model and tokenizer into GPU memory.

        Args:
            None

        Returns:
            None — sets self.model and self.tokenizer as instance attributes.
        """
        import os
        import time
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        vol.reload()

        if not os.path.exists(os.path.join(MODEL_WEIGHTS_DIR, "config.json")):
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_WEIGHTS_DIR,
                token=os.environ["HF_TOKEN"],
            )
            vol.commit()

        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_WEIGHTS_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval().cuda()
        print(f"[model load] {time.time() - t0:.1f}s")

        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_WEIGHTS_DIR,
            trust_remote_code=True,
        )
        print(f"[tokenizer load] {time.time() - t1:.1f}s")

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """
        Purpose: Generate a text response from a plain text prompt using
        the model's chat template.

        Args:
            prompt (str): The user message to send to the model.
            max_new_tokens (int): Maximum number of tokens to generate.
            enable_thinking (bool): If True, the model first emits a
                <think>...</think> reasoning block before its response.
                Defaults to False for concise, directly-parseable output.
            do_sample (bool): If True, use sampling (with `temperature`)
                instead of greedy decoding. Useful for retries after a
                malformed response.
            temperature (float): Sampling temperature, used only when
                do_sample is True.

        Returns:
            str: The model's text response (including any <think> block
                if enable_thinking is True).
        """
        import time
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
        ).to("cuda")

        generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            generate_kwargs["temperature"] = temperature

        t0 = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(inputs, **generate_kwargs)
        print(f"[generate] {time.time() - t0:.1f}s")

        response_ids = output_ids[0][inputs.shape[1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    @modal.method()
    def summarize_chapter(
        self,
        chapter_text: str,
        chapter_number: int,
        max_retries: int = 3,
    ) -> dict:
        """
        Purpose: Summarize a single chapter into the structured format used by
        book_chapter_context.json, retrying with sampling if the model's response
        isn't valid JSON matching the expected schema.

        Args:
            chapter_text (str): Full text of the chapter to summarize.
            chapter_number (int): 1-indexed chapter number being summarized.
            max_retries (int): Maximum number of generation attempts before
                giving up.

        Returns:
            dict: Parsed chapter data with keys "summary", "characters",
                "key_events", "locations", "cultural_references".

        Raises:
            ValueError: If no valid JSON response is produced within
                max_retries attempts.
        """
        prompt = build_summarization_prompt(chapter_text, chapter_number)

        last_error = None
        for attempt in range(max_retries):
            raw = self.generate.local(
                prompt,
                max_new_tokens=6144,
                enable_thinking=False,
                do_sample=(attempt > 0),
            )
            try:
                return parse_chapter_json(raw)
            except ValueError as e:
                last_error = e
                print(f"[summarize_chapter] attempt {attempt + 1} failed: {e}\nRaw response:\n{raw}")

        raise ValueError(f"Failed to get valid JSON after {max_retries} attempts: {last_error}")


@app.local_entrypoint()
def test_model():
    """
    Purpose: Validate that the model loads correctly on its dedicated
    volume/image and responds sensibly to a simple text prompt.

    Args:
        None

    Returns:
        None — result is printed to stdout.

    Usage:
        modal run preprocess_books.py::test_model
    """
    summarizer = ChapterSummarizer()
    response = summarizer.generate.remote("In one sentence, what is the capital of France?")
    print(f"\nResponse: {response}")


@app.local_entrypoint()
def process_book(
    book_name: str = "crime_and_punishment",
    start_chapter: int = 1,
    end_chapter: int = None,
):
    """
    Purpose: Run the full preprocessing pipeline over a range of chapters,
    summarizing chapters in parallel (up to max_containers GPUs at once) and
    writing results to a book_chapter_context.json file as each chapter
    completes.

    Args:
        book_name (str): Book identifier (e.g. "crime_and_punishment").
        start_chapter (int): 1-indexed chapter to start from. Defaults to 1.
        end_chapter (int, optional): 1-indexed chapter to end at (inclusive).
            Defaults to the book's last chapter.

    Returns:
        None — results are written incrementally to
        books/<book_name>/book_chapter_context.json and printed to
        stdout as each chapter completes.

    Usage:
        modal run preprocess_books.py::process_book
        modal run preprocess_books.py::process_book --start-chapter 1 --end-chapter 2
    """
    from book_utils import get_current_chapter, get_book_metadata

    metadata = get_book_metadata(book_name)
    if end_chapter is None:
        end_chapter = metadata["total_chapters"]

    output_path = Path(__file__).parent / "books" / book_name / "book_chapter_context.json"

    if output_path.exists():
        results = json.loads(output_path.read_text())
        done_chapters = {entry["number"] for entry in results}
        print(f"Resuming from {output_path} — {len(done_chapters)} chapter(s) already done")
    else:
        results = []
        done_chapters = set()

    chapter_numbers = [n for n in range(start_chapter, end_chapter + 1) if n not in done_chapters]
    for chapter_number in done_chapters & set(range(start_chapter, end_chapter + 1)):
        print(f"Chapter {chapter_number}: already done, skipping")

    if not chapter_numbers:
        return

    chapter_texts = [get_current_chapter(book_name, n) for n in chapter_numbers]

    print(f"Summarizing {len(chapter_numbers)} chapter(s) in parallel: {chapter_numbers}")

    summarizer = ChapterSummarizer()
    for chapter_number, result in zip(
        chapter_numbers,
        summarizer.summarize_chapter.map(chapter_texts, chapter_numbers),
    ):
        entry = {"number": chapter_number, **result}
        results.append(entry)
        results.sort(key=lambda e: e["number"])

        output_path.write_text(json.dumps(results, indent=2))
        print(f"Chapter {chapter_number}: done, wrote {output_path}")


@app.local_entrypoint()
def test_summarize_chapter(chapter_number: int = 1):
    """
    Purpose: Validate the summarization prompt and JSON parsing on a single
    chapter of "Crime and Punishment", for comparison against the
    hand-written reference in summary_key_data.json.

    Args:
        chapter_number (int): 1-indexed chapter number to summarize.
            Defaults to 1.

    Returns:
        None — the parsed JSON result is printed to stdout.

    Usage:
        modal run preprocess_books.py::test_summarize_chapter
        modal run preprocess_books.py::test_summarize_chapter --chapter-number 2
    """
    from book_utils import get_current_chapter

    chapter_text = get_current_chapter("crime_and_punishment", chapter_number)

    prompt = build_summarization_prompt(chapter_text, chapter_number)
    debug_path = Path(__file__).parent / "debug_prompt.txt"
    debug_path.write_text(prompt)
    print(f"Prompt written to {debug_path} ({len(prompt)} chars)")

    summarizer = ChapterSummarizer()
    result = summarizer.summarize_chapter.remote(
        chapter_text=chapter_text,
        chapter_number=chapter_number,
    )

    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def canonicalize_characters(book_name: str = "crime_and_punishment", max_retries: int = 3):
    """
    Purpose: Post-processing pass over book_chapter_context.json that
    resolves character name variants (e.g. "Raskolnikov" vs "Rodion
    Romanovitch Raskolnikov") into a single canonical name per person, then
    builds a book-wide character glossary (characters.json) with the chapter
    each character is first introduced.

    Args:
        book_name (str): Book identifier (e.g. "crime_and_punishment").
        max_retries (int): Maximum number of generation attempts before
            giving up on the canonicalization mapping.

    Returns:
        None — rewrites books/<book_name>/book_chapter_context.json
        with canonicalized character names/aliases, and writes
        books/<book_name>/characters.json.

    Usage:
        modal run preprocess_books.py::canonicalize_characters
    """
    book_dir = Path(__file__).parent / "books" / book_name
    summary_path = book_dir / "book_chapter_context.json"
    data = json.loads(summary_path.read_text())

    character_map = _collect_character_names(data)
    print(f"Found {len(character_map)} unique character name(s) across {len(data)} chapter(s)")

    prompt = build_canonicalization_prompt(character_map)
    expected_names = set(character_map.keys())

    summarizer = ChapterSummarizer()
    last_error = None
    mapping = None
    for attempt in range(max_retries):
        raw = summarizer.generate.remote(
            prompt,
            max_new_tokens=4096,
            enable_thinking=False,
            do_sample=(attempt > 0),
        )
        try:
            mapping = parse_canonicalization_json(raw, expected_names)
            break
        except ValueError as e:
            last_error = e
            print(f"[canonicalize_characters] attempt {attempt + 1} failed: {e}\nRaw response:\n{raw}")

    if mapping is None:
        raise ValueError(f"Failed to get valid canonicalization mapping after {max_retries} attempts: {last_error}")

    renamed = {name: canonical for name, canonical in mapping.items() if name != canonical}
    print(f"Canonicalized {len(renamed)} name variant(s): {renamed}")

    data = apply_canonicalization(data, mapping)
    data = merge_near_duplicate_names(data)
    data = accumulate_aliases(data)
    summary_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote canonicalized characters to {summary_path}")

    glossary = build_character_glossary(data)
    glossary_path = book_dir / "characters.json"
    glossary_path.write_text(json.dumps(glossary, indent=2))
    print(f"Wrote {len(glossary)} character(s) to {glossary_path}")
