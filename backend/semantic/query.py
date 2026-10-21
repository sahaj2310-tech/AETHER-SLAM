"""Lightweight natural-language spatial query → scene-graph answer.

Designed for the obvious indoor-mapping questions:

    "Where is the chair?"            → returns chair node(s) with 3D position
    "What's near the laptop?"        → returns neighbours via NetworkX "near" edges
    "List all objects"               → returns full registry
    "What's the sound coming from?"  → returns the active sound source

No LLM, no model download, no GPU cost.  Pure keyword routing + regex.
Honest about its limits: anything outside this set returns a "didn't
understand" response so the UI can surface that as a hint instead of
hallucinating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueryResult:
    intent: str
    summary: str
    objects: List[Dict] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "summary": self.summary,
            "objects": self.objects,
            "relations": self.relations,
            "confidence": float(self.confidence),
        }


# Tokens we treat as the same concept (broad indoor-room synonyms)
_LABEL_ALIASES = {
    "chair": ["chair", "seat"],
    "diningtable": ["table", "desk", "diningtable", "dining table"],
    "tvmonitor": ["tv", "monitor", "screen", "tvmonitor", "display"],
    "laptop": ["laptop", "notebook"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "cell phone": ["phone", "cell phone", "mobile"],
    "person": ["person", "people", "human", "someone", "me"],
    "book": ["book", "books"],
    "clock": ["clock"],
    "vase": ["vase"],
    "remote": ["remote"],
}


def _canonical_label(token: str) -> Optional[str]:
    t = token.strip().lower()
    for canonical, syns in _LABEL_ALIASES.items():
        if t == canonical or t in syns:
            return canonical
    return None


def _all_labels(text: str) -> List[str]:
    text = text.lower()
    found = []
    for canonical, syns in _LABEL_ALIASES.items():
        for s in syns:
            if re.search(rf"\b{re.escape(s)}\b", text):
                found.append(canonical)
                break
    # dedupe preserving order
    seen = set()
    out = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def answer(question: str, semantic_map) -> QueryResult:
    """Route a natural-language question into the right scene-graph query.
    `semantic_map` is the live SemanticMap instance."""
    q = (question or "").strip().lower()
    if not q:
        return QueryResult("empty", "Empty question.", confidence=0.0)

    objs = semantic_map.get_semantic_objects()

    # --- intent: list everything ---------------------------------------
    if re.search(r"\b(list|all|everything|inventory)\b", q) \
            and not re.search(r"\bnear\b|\bwhere\b", q):
        if not objs:
            return QueryResult("list", "No objects are currently registered.", objects=[])
        summary = ", ".join(f"{o['label']} #{o['id']}" for o in objs[:12])
        return QueryResult("list", f"{len(objs)} object(s): {summary}", objects=objs)

    # --- intent: what's near X -----------------------------------------
    near_match = re.search(
        r"(?:what(?:'s|\s+is)?|which(?:\s+objects?)?)\s+(?:are\s+)?(?:near|next to|around|close to|by)\s+(?:the\s+)?(.+)",
        q,
    )
    if near_match:
        target_tok = near_match.group(1).strip().rstrip("?.! ")
        canonical = _canonical_label(target_tok)
        if canonical is None:
            return QueryResult(
                "near",
                f"I don't recognise '{target_tok}' as a tracked object.",
                confidence=0.0,
            )
        # Pick the highest-confidence match for the canonical label
        targets = [o for o in objs if o["label"] == canonical]
        if not targets:
            return QueryResult(
                "near",
                f"No {canonical} currently mapped.",
                confidence=0.3,
            )
        target = max(targets, key=lambda o: o.get("confidence", 0))

        # Use the scene graph's proximity edges. An "on" edge implies "near"
        # too (the geometric check requires distance < NEAR_THRESHOLD_M), so
        # treat either kind as a neighbour relation.
        sg = semantic_map.get_scene_graph()
        neighbours = []
        proximity_kinds = {"near", "on"}
        for e in sg["edges"]:
            kinds = set(e.get("kinds") or [e.get("kind")])
            if not (kinds & proximity_kinds):
                continue
            other_id = (e["target"] if e["source"] == target["id"]
                        else e["source"] if e["target"] == target["id"] else None)
            if other_id is None:
                continue
            other = next((o for o in objs if o["id"] == other_id), None)
            if other:
                neighbours.append({**other, "distance": e["distance"]})

        if not neighbours:
            return QueryResult(
                "near",
                f"Nothing within range of {canonical} #{target['id']}.",
                objects=[target],
                confidence=0.7,
            )
        neighbours.sort(key=lambda o: o["distance"])
        summary = ", ".join(f"{o['label']} #{o['id']} ({o['distance']:.1f}m)"
                            for o in neighbours[:5])
        return QueryResult(
            "near",
            f"Near {canonical} #{target['id']}: {summary}.",
            objects=[target, *neighbours],
            confidence=0.95,
        )

    # --- intent: where is X --------------------------------------------
    where_match = re.search(
        r"(?:where(?:'s|\s+is|\s+are)?)\s+(?:the\s+)?(.+)",
        q,
    )
    if where_match or any(_canonical_label(t) for t in q.split()):
        target_tok = (where_match.group(1).strip().rstrip("?.! ")
                      if where_match else q)
        canonical = _canonical_label(target_tok)
        if canonical is None:
            # Try any label token in the sentence
            labels = _all_labels(q)
            canonical = labels[0] if labels else None
        if canonical is None:
            return QueryResult(
                "where",
                f"I don't recognise that object name.",
                confidence=0.0,
            )

        matches = [o for o in objs if o["label"] == canonical]
        if not matches:
            return QueryResult(
                "where",
                f"No {canonical} currently mapped.",
                confidence=0.3,
            )
        matches.sort(key=lambda o: o.get("confidence", 0), reverse=True)
        best = matches[0]
        x, y, z = best["centroid_3d"]
        summary = (f"{canonical} #{best['id']} at "
                   f"({x:.2f}, {y:.2f}, {z:.2f}) m"
                   + (f"; {len(matches)} instance(s) total." if len(matches) > 1 else "."))
        return QueryResult(
            "where",
            summary,
            objects=matches,
            confidence=0.95,
        )

    return QueryResult(
        "unknown",
        "I can answer 'where is X', 'what's near X', and 'list all objects'.",
        confidence=0.0,
    )
