"""Decompose a natural-language search into structured filters + a visual query.

SigLIP has no notion of *who* "Stefano" is — personal identity is the face
pipeline's job, not the vision model's — so a query like "Stefano eating food"
must not be fed wholesale to the image encoder. Instead we **decompose** it:

* known **person names** (from the ``persons`` table) are peeled off into a
  person filter (and the existing People AND-mode when more than one is named);
* coarse **count phrases** ("alone", "with other people", "in a group") map to
  the existing number-of-people buckets;
* whatever **residual text** remains ("eating food", "at the beach") is the
  visual query handed to SigLIP.

The structured legs are then AND-ed with the visual ranking by the search layer,
so "Stefano eating food" becomes *a photo containing Stefano that looks like
eating food*. When the named person has region embeddings, the API grounds the
residual on *their* region (``search.grounded_search``) so the score is about the
person rather than the whole frame; for an unnamed/un-embedded query it falls back
to the whole-image score (so it still means "containing").

This module is deliberately a small, dependency-free heuristic planner so it is
fully unit-testable without any model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Count-phrase → number-of-people buckets (see :data:`search.PEOPLE_BUCKETS`).
#: "alone" is a single-person (portrait) shot; "with others / group" is 2+.
_GROUP_RE = re.compile(
    r"\b(with (other |more )?(people|friends|others|family|guests)"
    r"|in a group|group (photo|picture|shot|of)|crowd|together)\b",
    re.IGNORECASE,
)
_ALONE_RE = re.compile(
    r"\b(alone|solo|by (him|her|my|them)self|by themselves|on (my|his|her|their) own"
    r"|just (me|him|her|them))\b",
    re.IGNORECASE,
)

#: A leading "a photo of / pictures showing / pics of" preamble carries no visual
#: signal of its own; strip it so the residual is the actual subject.
_PREAMBLE_RE = re.compile(
    r"^\s*(a |an |the )?(photos?|pictures?|pics?|images?|shots?|snaps?)\s+"
    r"(of|showing|with|containing)\s+",
    re.IGNORECASE,
)

#: Pure connective words trimmed from the residual's edges (kept internally so
#: e.g. "cup of coffee" survives).
_EDGE_WORDS = {
    "with", "and", "of", "the", "a", "an", "in", "on", "at", "is", "are",
    "to", "for", "&",
}


@dataclass
class QueryPlan:
    """The structured + visual legs a query decomposed into."""

    #: Matched person ids, in query order.
    person_ids: list[int] = field(default_factory=list)
    #: Matched person names (parallel to ``person_ids``), for display.
    person_names: list[str] = field(default_factory=list)
    #: ``"all"`` when 2+ people were named (AND them), else ``None``.
    person_mode: str | None = None
    #: Number-of-people bucket tokens (e.g. ``["1"]`` or ``["2-4", "5+"]``).
    people: list[str] = field(default_factory=list)
    #: The residual visual query for SigLIP (``""`` when nothing's left).
    text: str = ""

    @property
    def is_structured_only(self) -> bool:
        """True when the query reduced to filters with no visual residual."""

        return not self.text and bool(self.person_ids or self.people)


def _clean_residual(text: str) -> str:
    text = _PREAMBLE_RE.sub("", text)
    # Collapse the gaps left by removed spans, then trim connective edge words.
    tokens = text.split()
    lowered = [t.strip(",.;:!?").lower() for t in tokens]
    start, end = 0, len(tokens)
    while start < end and (not lowered[start] or lowered[start] in _EDGE_WORDS):
        start += 1
    while end > start and (not lowered[end - 1] or lowered[end - 1] in _EDGE_WORDS):
        end -= 1
    return " ".join(tokens[start:end]).strip(" ,.;:!?")


def plan_query(query: str, persons: list[dict]) -> QueryPlan:
    """Decompose ``query`` against the known ``persons`` (``{id, name}`` dicts).

    Returns a :class:`QueryPlan`. Person names are matched as whole words,
    case-insensitively, longest name first (so "Anna Maria" wins over "Anna").
    """

    working = query or ""
    plan = QueryPlan()

    # 1. Peel off known person names (longest first to prefer multi-word names).
    matched: list[tuple[int, str, int]] = []  # (id, name, position in query)
    for person in sorted(persons, key=lambda p: len(str(p.get("name") or "")), reverse=True):
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        m = pat.search(working)
        if m:
            matched.append((int(person["id"]), name, m.start()))
            working = pat.sub(" ", working)
    for pid, name, _ in sorted(matched, key=lambda t: t[2]):  # restore query order
        plan.person_ids.append(pid)
        plan.person_names.append(name)
    if len(plan.person_ids) >= 2:
        plan.person_mode = "all"  # "A and B" -> photos containing every one

    # 2. Coarse count phrases -> number-of-people buckets.
    if _GROUP_RE.search(working):
        plan.people = ["2-4", "5+"]
        working = _GROUP_RE.sub(" ", working)
    elif _ALONE_RE.search(working):
        plan.people = ["1"]
        working = _ALONE_RE.sub(" ", working)

    # 3. Whatever's left is the visual query.
    plan.text = _clean_residual(working)
    return plan
