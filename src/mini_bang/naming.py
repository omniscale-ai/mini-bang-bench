"""Readable durable run/work identities; timestamps are UTC with microseconds.

An identity is `<UTC time>--<role>[--<task>][--<instance>]`, for example
`YYYY-MM-DD-hh-mm-ss-ffffff--run--my-task--balanced-training`.
It must fit MAX_ID_LENGTH: the observation journal and scoring recovery accept
128 characters, and pool work names add a prefix within their 200-character limit.
When the labels do not fit, the shared middle words of the longest label are
replaced by one "_". Its first words (what is evaluated) and its last words
(the instance variant, e.g. `held-out-cases`) are kept.
"""
from datetime import datetime, timezone
import re

MAX_ID_LENGTH = 128
ELISION = "_"


def readable_slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "entry"


def _label(value, *, keep_dots=False):
    # "_" marks elision and "--" separates parts; dots of instance versions are kept.
    text = re.sub(r"[^a-z0-9.-]+" if keep_dots else r"[^a-z0-9-]+", "-", str(value).lower())
    return re.sub(r"-{2,}", "-", re.sub(r"\.{2,}", ".", text)).strip("-.")


def shorten_label(label, limit):
    """Keep the first and last words of a label; replace middle words by "_"."""
    if len(label) <= limit:
        return label
    words = re.findall(r"[^.-]+[.-]*", label)
    room = limit - len(ELISION)
    count, head = 0, ""
    while count < len(words) - 1 and len(head + words[count]) <= room // 4:
        head, count = head + words[count], count + 1
    head, count = (head.rstrip(".-"), count) if count else (words[0][: room // 4].rstrip(".-"), 1)
    tail = ""
    for word in reversed(words[count:]):
        if len(head) + len(word + tail) > room:
            break
        tail = word + tail
    tail = tail or words[-1][len(head) - room:]
    return head + ELISION + tail.lstrip(".-")


def readable_id(role, task=None, instance=None):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    role = _label(role)[:16].strip("-") or "work"
    task_label = (_label(task) or "task") if task else None
    instance_label = (_label(instance, keep_dots=True) or "instance") if instance else None
    labels = [label for label in (task_label, instance_label) if label]
    room = MAX_ID_LENGTH - len(stamp) - len(role) - 2 - 2 * len(labels)
    if task_label and instance_label and len(task_label) + len(instance_label) > room:
        instance_label = shorten_label(instance_label, max(room - len(task_label), room // 2))
    if task_label:
        task_label = shorten_label(task_label, room - len(instance_label or ""))
    elif instance_label:
        instance_label = shorten_label(instance_label, room)
    return "--".join(part for part in (stamp, role, task_label, instance_label) if part)
