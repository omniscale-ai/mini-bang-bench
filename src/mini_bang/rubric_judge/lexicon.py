"""Optional additional-language vocabulary for the rubric judge validators and prompts.

``lexicon.json`` next to this module holds every regex alternative, word
list and prompt example of that language. Without the file the validators
import normally and use their English vocabulary only. A list entry written
as ``@name`` stands for the top-level string entry ``name``.
"""
import json
from pathlib import Path

PATH = Path(__file__).with_name("lexicon.json")


def _load():
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


ENTRIES = _load()


def _lookup(path, default):
    value = ENTRIES
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def items(*path):
    """Entries of one list, or an empty list without the lexicon."""
    return [ENTRIES[value[1:]] if isinstance(value, str) and value.startswith("@") else value
            for value in _lookup(path, [])]


def alternatives(*path):
    """Regex alternatives to append to an alternation as '|a|b', or ''."""
    return "".join("|" + value for value in items(*path))


def words(*path, base=""):
    """Space-separated words: ``base`` followed by the lexicon words."""
    return " ".join(value for value in (base, *items(*path)) if value)


def text(*path, default=""):
    """One string entry, or ``default`` without the lexicon."""
    return _lookup(path, default)


def mapping(*path):
    """One object entry as a dict, or an empty dict without the lexicon."""
    return dict(_lookup(path, {}))


def fold(value):
    """Replace letter variants by their base letters."""
    for source, target in mapping("letter_folding").items():
        value = value.replace(source, target)
    return value
