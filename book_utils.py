#!/usr/bin/env python3
"""
Book utilities for Reading Buddy voice companion app.

This module provides functions to load book chapters and build spoiler-free
reading context for the AI voice model. The spoiler prevention is architectural:
the model only receives text up to the reader's current chapter.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Project root and books directory
PROJECT_ROOT = Path(__file__).parent
BOOKS_DIR = PROJECT_ROOT / 'books'

# Available book identifiers with metadata
AVAILABLE_BOOKS = {
    'crime_and_punishment': {
        'title': 'Crime and Punishment',
        'author': 'Fyodor Dostoevsky',
        'folder': 'crime_and_punishment',
        'description': '39 chapters across 6 parts'
    },
    'the_idiot': {
        'title': 'The Idiot',
        'author': 'Fyodor Dostoevsky',
        'folder': 'the_idiot',
        'description': '51 chapters across 4 parts'
    },
    'the_count_of_monte_cristo': {
        'title': 'The Count of Monte Cristo',
        'author': 'Alexandre Dumas',
        'folder': 'the_count_of_monte_cristo',
        'description': '117 chapters across 5 volumes'
    },
    'the_iliad': {
        'title': 'The Iliad',
        'author': 'Homer',
        'folder': 'the_iliad',
        'description': '24 books of epic poetry'
    }
}


def _book_dir(book: str) -> Path:
    """
    Get the directory containing a book's data files, validating that the
    book identifier is recognized.

    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').

    Returns:
        Path: Directory containing the book's chapters.json, book_chapter_context.json, etc.

    Raises:
        ValueError: If the book identifier is not recognized.
    """
    if book not in AVAILABLE_BOOKS:
        available = ', '.join(AVAILABLE_BOOKS.keys())
        raise ValueError(
            f"Unknown book: '{book}'. Available books: {available}"
        )

    return BOOKS_DIR / AVAILABLE_BOOKS[book]['folder']


def _load_chapters(book: str) -> List[Dict[str, Any]]:
    """
    Load chapters.json for a given book.

    Internal helper function that reads and parses the chapters.json file
    for the specified book. Validates that the book exists and the file
    can be loaded.

    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').

    Returns:
        list[dict]: List of chapter dictionaries with 'number', 'title', 'text' keys.

    Raises:
        ValueError: If the book identifier is not recognized.
        FileNotFoundError: If the chapters.json file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    chapters_path = _book_dir(book) / 'chapters.json'

    if not chapters_path.exists():
        raise FileNotFoundError(
            f"Chapters file not found: {chapters_path}. "
            f"Run parse_books.py to generate it."
        )
    
    with open(chapters_path, 'r', encoding='utf-8') as f:
        chapters = json.load(f)
    
    return chapters


def get_reading_context(book: str, chapter_number: int, max_chars: Optional[int] = 30000) -> str:
    """
    Get concatenated text of recent chapters up to and including the specified chapter.

    This function provides the reading context for the AI model, containing recent
    story events leading up to the reader's current position. This is part of the
    architectural foundation of spoiler prevention: the model never receives text
    from chapters beyond chapter_number.

    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').
        chapter_number (int): Current chapter number (1-indexed). Chapters from
            1 to this number (inclusive) are eligible for inclusion in the context.
        max_chars (int, optional): Maximum length of the returned context, in
            characters. If the concatenated text of chapters 1..chapter_number
            exceeds this, only the most recent (tail) portion is returned, so the
            prompt stays within the model's memory-safe input length. Pass None
            for no cap.

    Returns:
        str: Concatenated text of chapters up to chapter_number, with each chapter
            separated by double newlines, trimmed to at most max_chars characters
            (keeping the most recent text). Returns empty string if chapter_number < 1.

    Raises:
        ValueError: If book is not recognized or chapter_number exceeds total chapters.
        FileNotFoundError: If the book's chapters.json doesn't exist.

    Example:
        >>> context = get_reading_context('crime_and_punishment', 5)
        >>> # Returns text of chapters 1-5 concatenated together, trimmed to recent text
    """
    chapters = _load_chapters(book)

    if chapter_number > len(chapters):
        raise ValueError(
            f"Chapter {chapter_number} exceeds total chapters ({len(chapters)}) "
            f"for '{book}'"
        )

    if chapter_number < 1:
        return ""

    # Get chapters 1 through chapter_number (inclusive)
    relevant_chapters = [ch for ch in chapters if ch['number'] <= chapter_number]

    # Concatenate all chapter texts
    context_text = '\n\n'.join(ch['text'] for ch in relevant_chapters)

    # Keep only the most recent text so the prompt stays within the model's
    # memory-safe input length, while still never including future chapters.
    if max_chars is not None and len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    return context_text


def get_current_chapter(book: str, chapter_number: int) -> str:
    """
    Get the text of a specific chapter.
    
    Retrieves only the text content of the requested chapter, without any
    surrounding chapters. Useful for focused questions about the current chapter.
    
    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').
        chapter_number (int): Chapter number to retrieve (1-indexed).
    
    Returns:
        str: Text content of the specified chapter.
    
    Raises:
        ValueError: If book is not recognized or chapter_number is invalid.
        FileNotFoundError: If the book's chapters.json doesn't exist.
    
    Example:
        >>> chapter_text = get_current_chapter('crime_and_punishment', 3)
        >>> # Returns only the text of chapter 3
    """
    chapters = _load_chapters(book)
    
    if chapter_number < 1 or chapter_number > len(chapters):
        raise ValueError(
            f"Invalid chapter number {chapter_number}. "
            f"'{book}' has {len(chapters)} chapters (1-{len(chapters)})"
        )
    
    # Find the chapter with matching number
    for ch in chapters:
        if ch['number'] == chapter_number:
            return ch['text']
    
    # Should never reach here if chapters are numbered correctly
    raise ValueError(f"Chapter {chapter_number} not found in '{book}'")


def get_book_metadata(book: str) -> Dict[str, Any]:
    """
    Get metadata about a book including title, author, and chapter count.
    
    Useful for displaying book information in the app and validating chapter
    numbers before making queries.
    
    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').
    
    Returns:
        dict: Dictionary containing:
            - title (str): Full book title
            - author (str): Author name
            - total_chapters (int): Number of chapters in the book
            - book_id (str): The book identifier (same as input)
    """
    if book not in AVAILABLE_BOOKS:
        available = ', '.join(AVAILABLE_BOOKS.keys())
        raise ValueError(
            f"Unknown book: '{book}'. Available books: {available}"
        )
    
    chapters = _load_chapters(book)
    book_info = AVAILABLE_BOOKS[book]
    
    return {
        'book_id': book,
        'title': book_info['title'],
        'author': book_info['author'],
        'total_chapters': len(chapters)
    }


def list_books() -> List[Dict[str, str]]:
    """
    List all available books with their metadata.
    
    Returns basic information about all books in the system, useful for
    displaying a book selection menu in the app.
    
    Returns:
        list[dict]: List of book dictionaries, each containing:
            - book_id (str): Book identifier for use with other functions
            - title (str): Full book title
            - author (str): Author name
    
    Example:
        >>> books = list_books()
        >>> for book in books:
        ...     print(f"{book['title']} by {book['author']}")
    """
    return [
        {
            'book_id': book_id,
            'title': info['title'],
            'author': info['author']
        }
        for book_id, info in AVAILABLE_BOOKS.items()
    ]


def describe_reading_context(book: str, chapter_number: int) -> tuple:
    """
    Get the reading context (raw chapter text) for a book up to the given
    chapter, along with a description of where it came from.

    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').
        chapter_number (int): Current chapter number (1-indexed).

    Returns:
        tuple[str, str]:
            context (str): Text of chapters up to chapter_number, as
                returned by get_reading_context.
            source_label (str): Description of the context's source, e.g.
                'the novel "Crime and Punishment" by Fyodor Dostoevsky
                (up to chapter 7 of 39)'.

    Raises:
        ValueError: If book is not recognized or chapter_number is invalid.
        FileNotFoundError: If the book's chapters.json doesn't exist.
    """
    metadata = get_book_metadata(book)
    context = get_reading_context(book, chapter_number)

    source_label = (
        f"the novel \"{metadata['title']}\" by {metadata['author']} "
        f"(up to chapter {chapter_number} of {metadata['total_chapters']})"
    )

    return context, source_label


def load_hybrid_context(book: str, chapter_numbers: List[int]) -> str:
    """
    Build a context string from a book's book_chapter_context.json file,
    combining per-chapter prose summaries with a deduplicated structured
    block of characters, key events, locations, and cultural references
    for the requested chapters.

    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').
        chapter_numbers (list[int]): Chapter numbers to include (e.g. [1, 2]).

    Returns:
        str: Formatted context string ready to pass to build_system_prompt.

    Raises:
        ValueError: If the book identifier is not recognized.
        FileNotFoundError: If the book's book_chapter_context.json doesn't exist.
    """
    summary_path = _book_dir(book) / 'book_chapter_context.json'

    with open(summary_path, 'r', encoding='utf-8') as f:
        all_chapters = json.load(f)

    chapters = [ch for ch in all_chapters if ch['number'] in chapter_numbers]

    sections = ["SUMMARY:"]
    for ch in chapters:
        sections.append(f"\nChapter {ch['number']}: {ch['summary']}")

    characters = {}
    key_events = []
    locations = []
    cultural_references = {}
    for ch in chapters:
        for c in ch['characters']:
            characters[c['name']] = c
        key_events.extend(ch['key_events'])
        locations.extend(ch['locations'])
        for cr in ch['cultural_references']:
            cultural_references[cr['term']] = cr

    sections.append("\n\nCHARACTERS:")
    for c in characters.values():
        aliases = f" (also: {', '.join(c['aliases'])})" if c['aliases'] else ""
        sections.append(f"- {c['name']}{aliases}: {c['description']}")

    sections.append("\nKEY EVENTS:")
    for event in key_events:
        sections.append(f"- {event}")

    sections.append("\nLOCATIONS:")
    for loc in dict.fromkeys(locations):
        sections.append(f"- {loc}")

    sections.append("\nCULTURAL/HISTORICAL NOTES:")
    for cr in cultural_references.values():
        sections.append(f"- {cr['term']}: {cr['explanation']}")

    return "\n".join(sections)


def describe_hybrid_context(book: str, chapter_numbers: List[int]) -> tuple:
    """
    Get a hybrid (summary + structured data) context for a book's chapters,
    along with a description of where it came from.

    Args:
        book (str): Book identifier (e.g., 'crime_and_punishment').
        chapter_numbers (list[int]): Chapter numbers to include (e.g. [1, 2]).

    Returns:
        tuple[str, str]:
            context (str): Hybrid context, as returned by load_hybrid_context.
            source_label (str): Description of the context's source, e.g.
                'a summary of chapters 1-2 of "Crime and Punishment" by
                Fyodor Dostoevsky'.

    Raises:
        ValueError: If the book identifier is not recognized.
        FileNotFoundError: If the book's book_chapter_context.json doesn't exist.
    """
    metadata = get_book_metadata(book)
    context = load_hybrid_context(book, chapter_numbers)

    chapter_range = (
        str(chapter_numbers[0])
        if len(chapter_numbers) == 1
        else f"{chapter_numbers[0]}-{chapter_numbers[-1]}"
    )
    source_label = (
        f"a summary of chapters {chapter_range} of "
        f"\"{metadata['title']}\" by {metadata['author']}"
    )

    return context, source_label
