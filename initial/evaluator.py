"""
Evaluator for anonymization clients.

- Loads a dataset in RONEC-style JSON format (list of dicts with keys: tokens, ner_tags, space_after)
- Reconstructs raw text and gold entity spans with character offsets
- Runs an anonymization client to get predicted spans via metadata
- Computes micro precision/recall/F1 (optionally span-only with --ignore-labels)
- Verifies deanonymization fidelity (exact text match to original)

Usage examples:

CLI (default mock client):
  python evaluator.py --data mock_subset_200.json --limit 200

Programmatic:
  from anonymizer_mock import AnonymizerMock
  from evaluator import Evaluator, load_dataset

  client = AnonymizerMock()
  examples = load_dataset("mock_subset_200.json", limit=200)
  evalr = Evaluator(client, ignore_labels=False)
  metrics = evalr.evaluate(examples)
  print(metrics)
"""
from typing import List, Dict, Tuple, Any
import argparse



def detokenize_with_offsets(tokens: List[str], space_after: List[bool]) -> Tuple[str, List[Tuple[int, int]]]:
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


def bio2_to_spans(tokens: List[str], tags: List[str], space_after: List[bool]) -> List[Dict[str, Any]]:
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
        examples.append({
            "text": text,
            "gold_spans": gold_spans,
            "tokens": tokens,
            "tags": tags,
            "space_after": space_after,
        })
        if limit is not None and len(examples) >= limit:
            break
    return examples


class Evaluator:
    def __init__(self, client: Any, ignore_labels: bool = False):
        self.client = client
        self.ignore_labels = ignore_labels

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
        deanonym_ok = 0
        total = 0

        for ex in examples:
            total += 1
            text = ex["text"]
            gold = ex["gold_spans"]
            anon_text, metadata = self.client.anonymize(text)

            # Predicted spans expected in metadata["entities"]
            pred_spans = metadata.get("entities", []) if isinstance(metadata, dict) else []

            # Deanonymization check
            try:
                deanon = self.client.deanonymize(anon_text, metadata)
                if deanon == text:
                    deanonym_ok += 1
            except Exception:
                pass

            gold_set = self._to_tuple_set(gold)
            pred_set = self._to_tuple_set(pred_spans)

            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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
    """CLI to run evaluator using a client implementation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="roner_eval.json", help="Path to RONEC-style JSON file")
    parser.add_argument("--limit", type=int, default=50, help="Max samples to evaluate (for speed)")
    # --ignore-labels is only used on ronec original data because they use different labels
    parser.add_argument("--ignore-labels", action="store_true",
                        help="If set, evaluate spans ignoring labels (useful when label taxonomies differ)")
    args = parser.parse_args()

    # Load data
    examples = load_dataset(args.data, limit=args.limit)
    if not examples:
        print("No examples loaded. Check the --data path.")
        return

    # Init client and evaluator, use your client here
    from anonymizer_ronec import AnonymizerRonec as Anonymizer
    # from anonymizer_mock import AnonymizerMock as Anonymizer
    # from anonymizer_template import Anonymizer

    client = Anonymizer()
    evaluator = Evaluator(client, ignore_labels=args.ignore_labels)

    # Evaluate
    metrics = evaluator.evaluate(examples)
    print("Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
