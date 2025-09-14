"""
Base interface and default client implementation for anonymization.

Participants only need to deliver the model (weights/checkpoint). The default
client (`MyAnonymizer`) uses the Hugging Face model
"dumitrescustefan/bert-base-romanian-ner" via Transformers pipeline.

Contract for metadata returned by anonymize:
{
  "entities": [
    {
      "start": int,          # start char offset in original text (inclusive)
      "end": int,            # end char offset in original text (exclusive)
      "label": str,          # entity label (e.g., PERSON, GPE, EMAIL, etc.)
      "text": str,           # original surface form
      "replacement": str     # placeholder used in anonymized text (e.g., <PERSON_1>)
    }, ...
  ]
}
This information must be sufficient for a perfect deanonymization.
"""
from typing import Tuple, Dict, List, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class AnonymizerRonec:
    """Default anonymizer using HF model dumitrescustefan/bert-base-romanian-ner."""

    def __init__(self):
        model_name = "dumitrescustefan/bert-base-romanian-ner"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

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

    def anonymize(self, text: str) -> Tuple[str, Dict]:
        preds = self.pipe(text)
        predicted_spans: List[Dict[str, Any]] = []
        for p in preds:
            s = int(p.get("start", 0))
            e = int(p.get("end", 0))
            word = p.get("word", "")
            group = p.get("entity_group", p.get("entity", ""))
            label = self._map_label(group)
            if e > s:
                predicted_spans.append({
                    "start": s,
                    "end": e,
                    "label": label,
                    "text": word if word else text[s:e],
                })

        predicted_spans.sort(key=lambda s: s["start"])  # left-to-right

        anonymized_parts: List[str] = []
        cursor = 0
        entities_meta: List[Dict[str, Any]] = []
        for idx, span in enumerate(predicted_spans, start=1):
            s, e, label = span["start"], span["end"], span["label"]
            placeholder = f"<{label}_{idx}>"
            anonymized_parts.append(text[cursor:s])
            anonymized_parts.append(placeholder)
            cursor = e
            entities_meta.append({
                "start": s,
                "end": e,
                "label": label,
                "text": text[s:e],
                "replacement": placeholder,
            })
        anonymized_parts.append(text[cursor:])
        anonymized_text = "".join(anonymized_parts)
        metadata = {"entities": entities_meta}
        return anonymized_text, metadata

    def deanonymize(self, text: str, metadata: Dict) -> str:
        entities = metadata.get("entities", []) if isinstance(metadata, dict) else []
        result = text
        for ent in reversed(entities):
            placeholder = ent["replacement"]
            original = ent["text"]
            result = result.replace(placeholder, original, 1)
        return result


