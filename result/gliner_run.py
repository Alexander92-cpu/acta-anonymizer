from typing import Tuple, Dict, List, Any
import torch
from torch.amp import autocast
from gliner import GLiNER

class AnonymizerGLiNER:

    def __init__(self, model_path: str = "gliner_Med", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GLiNER.from_pretrained(model_path, load_tokenizer=True, local_files_only=True)
        self.model.to(self.device)
        print(f"[GLiNER Anonymizer] Model loaded to {self.device}")

        self.label_map = {label: label for label in [
            "Full Name", "Personal Numeric Code (CNP)", "Date of Birth", "Gender", "Nationality",
            "Spoken Language", "Address", "Work Address", "Mobile Phone", "Landline Phone",
            "Email", "Postal Code", "Birth City", "Birth Country", "Profession", "Activity / Occupation",
            "Employer", "Income", "Marital Status", "Education", "IBAN", "Bank Account", "Card Number",
            "Passport", "Identity Card", "License Number", "Health Insurance", "Blood Type", "Allergies",
            "Medical Conditions", "IP Address", "Username", "Device ID", "Biometric Data", "Contract Number",
            "Plate Number", "Digital Account", "Crypto Wallet", "Alternate Account Number", "Segment",
            "Politically Exposed Person (PEP)", "FATCA Status"
        ]}

    def _map_label(self, label: str) -> str:
        return self.label_map.get(label, "MISC")

    def anonymize(self, text: str) -> Tuple[str, Dict]:
        with autocast(self.device):
            entities = self.model.predict_entities(text, list(self.label_map.keys()), threshold=0.3)

        parts: List[str] = []
        cursor = 0
        entities_meta: List[Dict[str, Any]] = []

        for idx, ent in enumerate(sorted(entities, key=lambda e: e["start"])):
            s, e, label = ent["start"], ent["end"], self._map_label(ent["label"])
            placeholder = f"<{label}_{idx+1}>"
            parts.append(text[cursor:s])
            parts.append(placeholder)
            cursor = e
            entities_meta.append({
                "start": s,
                "end": e,
                "label": label,
                "text": text[s:e],
                "replacement": placeholder
            })

        parts.append(text[cursor:])
        anon_text = "".join(parts)
        return anon_text, {"entities": entities_meta}

    def deanonymize(self, text: str, metadata: Dict) -> str:
        result = text
        for ent in reversed(metadata.get("entities", [])):
            result = result.replace(ent["replacement"], ent["text"], 1)
        return result
