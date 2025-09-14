import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import yaml

from inference import RomanianNERInference

# ----------------------
# Evaluator utilities
# ----------------------


def detokenize_with_offsets(
    tokens: List[str], space_after: List[bool]
) -> Tuple[str, List[Tuple[int, int]]]:
    """Detokenize using space_after and return text and per-token (start,end) char spans.
    end is exclusive.
    """
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


def bio2_to_spans(
    tokens: List[str], tags: List[str], space_after: List[bool]
) -> List[Dict[str, Any]]:
    """Convert BIO2 tags to character-level spans with labels and surface text.
    Returns list of dicts: {start, end, label, text}
    """
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
            spans.append(
                {
                    "start": start_char,
                    "end": end_char,
                    "label": label,
                    "text": surface,
                }
            )
            i = j
        else:
            i += 1
    return spans


def load_dataset(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load a RONEC-style JSON list and return a list of examples with
    fields: text, gold_spans, tokens, tags, space_after (for debugging if needed)
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples: List[Dict[str, Any]] = []
    for i, ex in enumerate(data):
        tokens = ex["tokens"]
        tags = ex["ner_tags"]
        space_after = ex["space_after"]
        text, _ = detokenize_with_offsets(tokens, space_after)
        gold_spans = bio2_to_spans(tokens, tags, space_after)
        examples.append(
            {
                "text": text,
                "gold_spans": gold_spans,
                "tokens": tokens,
                "tags": tags,
                "space_after": space_after,
            }
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


class Evaluator:
    def __init__(self, inference_config_path: str, ignore_labels: bool = False):
        self.infer = RomanianNERInference(config_path=inference_config_path)
        self.ignore_labels = ignore_labels

    @staticmethod
    def _map_label(entity_group: str) -> str:
        """Normalize entity group to evaluation label."""
        if not entity_group:
            return "MISC"
        eg = entity_group.upper()
        if eg == "PER":
            return "PERSON"
        if eg in ("ORG", "LOC"):
            return eg
        return eg

    def predict(self, text: str) -> Dict:
        preds = self.infer.pipe(text)
        predicted_spans: List[Dict[str, Any]] = []
        for p in preds:
            s = int(p.get("start", 0))
            e = int(p.get("end", 0))
            word = p.get("word", "")
            group = p.get("entity_group", p.get("entity", ""))
            label = self._map_label(group)
            if e > s:
                predicted_spans.append(
                    {
                        "start": s,
                        "end": e,
                        "label": label,
                        "text": word if word else text[s:e],
                    }
                )

        predicted_spans.sort(key=lambda s: s["start"])  # left-to-right

        entities_meta: List[Dict[str, Any]] = []
        for idx, span in enumerate(predicted_spans, start=1):
            s, e, label = span["start"], span["end"], span["label"]
            placeholder = f"<{label}_{idx}>"
            entities_meta.append(
                {
                    "start": s,
                    "end": e,
                    "label": label,
                    "text": text[s:e],
                    "replacement": placeholder,
                }
            )
        metadata = {"entities": entities_meta}
        return metadata

    def _to_tuple_set(self, spans: List[Dict[str, Any]]):
        if self.ignore_labels:
            return {(s["start"], s["end"]) for s in spans}
        return {(s["start"], s["end"], s["label"]) for s in spans}

    def evaluate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate micro P/R/F1 and deanonymization fidelity.
        examples: list of {text, gold_spans}
        """
        tp = 0
        fp = 0
        fn = 0
        total = 0

        for ex in examples:
            total += 1
            text = ex["text"]
            gold = ex["gold_spans"]
            metadata = self.predict(text)

            # Predicted spans expected in metadata["entities"]
            pred_spans = (
                metadata.get("entities", []) if isinstance(metadata, dict) else []
            )

            gold_set = self._to_tuple_set(gold)
            pred_set = self._to_tuple_set(pred_spans)

            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "samples": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def main():
    with open("configs/evaluation.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    examples = load_dataset(config["evaluation"]["data_path"])

    evaluator = Evaluator("configs/inference.yaml")

    metrics = evaluator.evaluate(examples)
    print("Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
