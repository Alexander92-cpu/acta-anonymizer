"""
Hackathon Anonymizer Template

Participants: copy this file, or keep the same file name, and only change
MODEL_NAME to your own model. Keep the same
anonymize/deanonymize signatures.

Requirements:
- Your model must be a Hugging Face token-classification model compatible with
  AutoTokenizer and AutoModelForTokenClassification.
- The code uses the Transformers pipeline with aggregation_strategy="simple"
  to produce character-level entity spans.

The evaluator expects anonymize(text) -> (anonymized_text, metadata) where
metadata["entities"] is a list of dicts with:
  { start, end, label, text, replacement }
so that deanonymize(anonymized_text, metadata) restores the original text.
"""
from typing import Tuple, Dict, List, Any, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# Replace this with your model name (Hugging Face hub) OR a local path to a
# fine-tuned Transformer model directory. If you trained locally and saved the
# model and tokenizer to a folder (with config.json, tokenizer files, etc.),
# set MODEL_NAME to that folder path. You can also set TOKENIZER_PATH separately
# if your tokenizer is stored in a different folder.
MODEL_PATH = None  # Set to a HF repo ID (e.g., "org/model") or a local folder with model files

# Optional: set only if your tokenizer is stored separately from the model.
# When to use:
# - If your local fine-tuned checkpoint folder already contains tokenizer files
#   (tokenizer.json, vocab files, merges.txt, etc.), LEAVE THIS AS None.
# - If you keep the tokenizer in a different folder than the model weights,
#   set TOKENIZER_PATH to that folder path.
# - If you use a Hugging Face repo ID for MODEL_NAME, LEAVE THIS AS None
#   (the tokenizer will be fetched from the same repo).
# Examples:
#   MODEL_NAME = "/path/to/model_dir"; TOKENIZER_PATH = None
#   MODEL_NAME = "/path/to/model_dir"; TOKENIZER_PATH = "/path/to/tokenizer_dir"
#   MODEL_NAME = "org/my-finetuned-model"; TOKENIZER_PATH = None
TOKENIZER_PATH: Optional[str] = None  # default: use MODEL_NAME

# Label map EXACTLY matching the Moldova-specific PII labels (do not change).
LABEL_MAP = {
    # Core Identity
    "NUME_PRENUME": "NUME_PRENUME",
    "CNP": "CNP",
    "DATA_NASTERII": "DATA_NASTERII",
    "SEX": "SEX",
    "NATIONALITATE": "NATIONALITATE",
    "LIMBA_VORBITA": "LIMBA_VORBITA",

    # Contact Information
    "ADRESA": "ADRESA",
    "ADRESA_LUCRU": "ADRESA_LUCRU",
    "TELEFON_MOBIL": "TELEFON_MOBIL",
    "TELEFON_FIX": "TELEFON_FIX",
    "EMAIL": "EMAIL",
    "COD_POSTAL": "COD_POSTAL",

    # Location & Origin
    "ORAS_NASTERE": "ORAS_NASTERE",
    "TARA_NASTERE": "TARA_NASTERE",

    # Professional Information
    "PROFESIE": "PROFESIE",
    "ACTIVITATE": "ACTIVITATE",
    "ANGAJATOR": "ANGAJATOR",
    "VENIT": "VENIT",

    # Personal Status
    "STARE_CIVILA": "STARE_CIVILA",
    "EDUCATIE": "EDUCATIE",

    # Financial Information
    "IBAN": "IBAN",
    "CONT_BANCAR": "CONT_BANCAR",
    "CARD_NUMBER": "CARD_NUMBER",

    # Identity Documents
    "PASAPORT": "PASAPORT",
    "BULETIN": "BULETIN",
    "NUMAR_LICENTA": "NUMAR_LICENTA",

    # Medical Information
    "ASIGURARE_MEDICALA": "ASIGURARE_MEDICALA",
    "GRUPA_SANGE": "GRUPA_SANGE",
    "ALERGII": "ALERGII",
    "CONDITII_MEDICALE": "CONDITII_MEDICALE",

    # Digital & Technology
    "IP_ADDRESS": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "DEVICE_ID": "DEVICE_ID",
    "BIOMETRIC": "BIOMETRIC",

    # Additional Financial & Legal
    "NUMAR_CONTRACT": "NUMAR_CONTRACT",
    "NUMAR_PLACA": "NUMAR_PLACA",
    "CONT_DIGITAL": "CONT_DIGITAL",
    "WALLET_CRYPTO": "WALLET_CRYPTO",
    "NUMAR_CONT_ALT": "NUMAR_CONT_ALT",

    # Other
    "SEGMENT": "SEGMENT",
    "EXPUS_POLITIC": "EXPUS_POLITIC",
    "STATUT_FATCA": "STATUT_FATCA",
}


class Anonymizer:
    """Minimal anonymizer that you can use as a starting point.
    Only change MODEL_NAME and (optionally) LABEL_MAP above.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 device: Optional[str] = None):
        name = model_path or MODEL_PATH
        tok_name = tokenizer_path or TOKENIZER_PATH or name

        # Robust tokenizer load (works for hub and local dirs; fall back to non-fast)
        if os.path.isdir(tok_name):
            print(f"[Anonymizer] Loading tokenizer from local path: {tok_name}")
        else:
            print(f"[Anonymizer] Loading tokenizer '{tok_name}' from Hugging Face hub (may download files, please wait)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=False)

        if os.path.isdir(name):
            print(f"[Anonymizer] Loading model from local path: {name}")
        else:
            print(f"[Anonymizer] Loading model '{name}' from Hugging Face hub (downloading weights if needed, this can take a few minutes)...")
        self.model = AutoModelForTokenClassification.from_pretrained(name)

        # Device selection: let pipeline auto-detect unless explicitly provided
        pipe_kwargs = {
            "task": "token-classification",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "aggregation_strategy": "simple",
        }
        if device:
            # Accept "cpu", "cuda", "cuda:0", "mps"
            pipe_kwargs["device"] = device

        self.pipe = pipeline(**pipe_kwargs)
        print("[Anonymizer] Model is ready.")

    @staticmethod
    def _map_label(entity_group: str) -> str:
        if not entity_group:
            return "MISC"
        key = entity_group.upper()
        return LABEL_MAP.get(key, key)

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

        # Build anonymized text with placeholders and metadata
        parts: List[str] = []
        cursor = 0
        entities_meta: List[Dict[str, Any]] = []
        for idx, span in enumerate(predicted_spans, start=1):
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
        metadata = {"entities": entities_meta}
        return anon_text, metadata

    def deanonymize(self, text: str, metadata: Dict) -> str:
        entities = metadata.get("entities", []) if isinstance(metadata, dict) else []
        result = text
        for ent in reversed(entities):
            result = result.replace(ent["replacement"], ent["text"], 1)
        return result
