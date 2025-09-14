from typing import Tuple, Dict, List, Any
import torch
from torch.amp import autocast
from gliner import GLiNER

from result.utils import post_process_entities


class AnonymizerGLiNER:
    def __init__(self, model_path: str = "gliner", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GLiNER.from_pretrained(model_path, load_tokenizer=True, local_files_only=True)
        self.model.to(self.device)
        print(f"model loaded to {self.device}")

        # self.model_labels = [
        #     "Full Name", "Personal Numeric Code (CNP)", "Date of Birth", "Gender", "Nationality",
        #     "Spoken Language", "Address", "Work Address", "Mobile Phone", "Landline Phone",
        #     "Email", "Postal Code", "Birth City", "Birth Country", "Profession", "Activity / Occupation",
        #     "Employer", "Income", "Marital Status", "Education", "IBAN", "Bank Account",
        #     "Card Number", "Passport", "Identity Card", "License Number", "Health Insurance",
        #     "Blood Type", "Allergies", "Medical Conditions", "IP Address", "Username",
        #     "Device ID", "Biometric Data", "Contract Number", "Plate Number",
        #     "Digital Account", "Crypto Wallet", "Alternate Account Number", "Segment",
        #     "Politically Exposed Person (PEP)", "FATCA Status"
        # ]
        self.model_labels = [
            "person", "cpf", "date of birth", "gender", "nationality",
            "spoken Language", "address", "work address", "mobile phone number", "landline phone number",
            "email", "postal code", "birth city", "birth country", "profession", "occupation",
            "employer", "income", "marital status", "education", "iban", "bank account number",
            "credit card number", "passport_number", "identity card number", "driver's license number", "health insurance number",
            "blood type", "allergies", "medical condition", "ip address", "username",
            "device identifier", "biometric data", "contract number", "license plate number",
            "digital wallet account", "cryptocurrency wallet address", "other account number", "Segment",
            "politically exposed person", "FATCA status"
        ]

        self.task_labels = ['NUME_PRENUME', 'CNP', 'DATA_NASTERII', 'SEX', 'NATIONALITATE', 'LIMBA_VORBITA', 'ADRESA', 'ADRESA_LUCRU', 'TELEFON_MOBIL', 'TELEFON_FIX', 'EMAIL', 'COD_POSTAL', 'ORAS_NASTERE', 'TARA_NASTERE', 'PROFESIE', 'ACTIVITATE', 'ANGAJATOR', 'VENIT', 'STARE_CIVILA', 'EDUCATIE', 'IBAN', 'CONT_BANCAR', 'CARD_NUMBER', 'PASAPORT', 'BULETIN', 'NUMAR_LICENTA', 'ASIGURARE_MEDICALA', 'GRUPA_SANGE', 'ALERGII', 'CONDITII_MEDICALE', 'IP_ADDRESS', 'USERNAME', 'DEVICE_ID', 'BIOMETRIC', 'NUMAR_CONTRACT', 'NUMAR_PLACA', 'CONT_DIGITAL', 'WALLET_CRYPTO', 'NUMAR_CONT_ALT', 'SEGMENT', 'EXPUS_POLITIC', 'STATUT_FATCA']

        self.eval_label_map = {model_label: task_label for model_label, task_label in zip(self.model_labels, self.task_labels)}


    def _map_label(self, label: str) -> str:
        # Map GLiNER prediction label to evaluation label
        return self.eval_label_map.get(label, "")

    def anonymize(self, text: str) -> Tuple[str, Dict]:
        with autocast(self.device):
            entities = self.model.predict_entities(text, self.model_labels, threshold=0.4)
        # entities = post_process_entities(text, entities)
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
