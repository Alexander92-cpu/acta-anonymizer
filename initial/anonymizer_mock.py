"""
Mock anonymizer that emulates an LLM with prior knowledge of entity placements
from a fixed subset of the synthetic dataset.

- Extracts the first 200 items from the source dataset into a subset file
- Builds a mapping from detokenized text to gold spans
- On anonymize(), returns gold spans with 95% success probability (per-entity)
- Deanonymize() restores placeholders back to original text
"""
import json
import os
import random
from typing import List, Dict, Any, Tuple

# Paths
SUBSET_PATH = "../data/mock_subset_200.json"


def detokenize_with_offsets(tokens: List[str], space_after: List[bool]) -> Tuple[str, List[Tuple[int, int]]]:
    text_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        start = cursor
        text_parts.append(tok)
        cursor += len(tok)
        end = cursor
        spans.append((start, end))
        if i < len(tokens) - 1 and space_after[i]:
            text_parts.append(" ")
            cursor += 1
    text = "".join(text_parts)
    return text, spans


def bio2_to_spans(tokens: List[str], tags: List[str], space_after: List[bool]) -> List[Dict[str, Any]]:
    text, tok_spans = detokenize_with_offsets(tokens, space_after)
    spans: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            start_char = tok_spans[i][0]
            j = i + 1
            while j < len(tokens) and tags[j].startswith("I-") and tags[j][2:] == label:
                j += 1
            end_char = tok_spans[j - 1][1]
            surface = text[start_char:end_char]
            spans.append({
                "start": start_char,
                "end": end_char,
                "label": label,
                "text": surface,
            })
            i = j
        else:
            i += 1
    return spans


class AnonymizerMock:
    def __init__(self):
        # Single knob:
        #  - true positive kept with probability (1 - error_rate)
        #  - false positive added with probability (error_rate) per gold span
        self.error_rate = 0.05
        self.success_prob = 1.0 - self.error_rate
        # Build mapping from detokenized text -> gold spans
        with open(SUBSET_PATH, "r", encoding="utf-8") as f:
            subset = json.load(f)
        self.gold_by_text: Dict[str, List[Dict[str, Any]]] = {}
        for ex in subset:
            tokens = ex["tokens"]
            tags = ex["ner_tags"]
            space_after = ex["space_after"]
            text, _ = detokenize_with_offsets(tokens, space_after)
            spans = bio2_to_spans(tokens, tags, space_after)
            self.gold_by_text[text] = spans

    def anonymize(self, text: str) -> Tuple[str, Dict[str, Any]]:
        gold_spans = self.gold_by_text.get(text, [])
        predicted: List[Dict[str, Any]] = []
        for s in gold_spans:
            if random.random() < self.success_prob:
                predicted.append({
                    "start": s["start"],
                    "end": s["end"],
                    "label": s["label"],
                    "text": s["text"],
                })
        # Add ~5% false positives relative to number of gold spans
        if gold_spans:
            labels = list({s["label"] for s in gold_spans}) or ["MISC"]
            text_len = len(text)

            def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
                return not (a[1] <= b[0] or b[1] <= a[0])

            occupied: List[Tuple[int, int]] = []
            # Occupied initially by predicted spans
            occupied.extend((p["start"], p["end"]) for p in predicted)
            # Also avoid overlapping with gold spans to keep things sane
            occupied.extend((g["start"], g["end"]) for g in gold_spans)

            for _ in gold_spans:
                if random.random() < self.error_rate and text_len >= 2:
                    # Try a few times to sample a non-overlapping fake span
                    for _attempt in range(20):
                        s = random.randint(0, max(0, text_len - 2))
                        max_len = min(12, text_len - s)
                        e = s + random.randint(1, max_len)
                        cand = (s, e)
                        if all(not overlaps(cand, occ) for occ in occupied):
                            label = random.choice(labels)
                            predicted.append({
                                "start": s,
                                "end": e,
                                "label": label,
                                "text": text[s:e],
                            })
                            occupied.append(cand)
                            break

        # Sort by start
        predicted.sort(key=lambda x: x["start"]) 
        # Build anonymized text with placeholders
        parts: List[str] = []
        cursor = 0
        entities_meta: List[Dict[str, Any]] = []
        for idx, span in enumerate(predicted, start=1):
            s, e, label = span["start"], span["end"], span["label"]
            placeholder = f"<{label}_{idx}>"
            parts.append(text[cursor:s])
            parts.append(placeholder)
            cursor = e
            entities_meta.append({
                "start": s,
                "end": e,
                "label": label,
                "text": text[s:e],
                "replacement": placeholder,
            })
        parts.append(text[cursor:])
        anon_text = "".join(parts)
        return anon_text, {"entities": entities_meta}

    def deanonymize(self, text: str, metadata: Dict[str, Any]) -> str:
        entities = metadata.get("entities", []) if isinstance(metadata, dict) else []
        result = text
        for ent in reversed(entities):
            result = result.replace(ent["replacement"], ent["text"], 1)
        return result
