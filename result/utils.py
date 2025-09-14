import re
from typing import List, Dict, Any

ROMANIAN_NAME_BLACKLIST = {"Dl.", "Dna.", "Dl", "Dna", "Dr.", "Prof.", "Ing.", "Av.", "Sr."}

def merge_overlapping_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping spans, keeping the longest span."""
    if not entities:
        return []
    entities = sorted(entities, key=lambda e: (e["start"], -e["end"]))
    merged: List[Dict[str, Any]] = [entities[0]]
    for ent in entities[1:]:
        last = merged[-1]
        if ent["start"] < last["end"]:
            if ent["end"] - ent["start"] > last["end"] - last["start"]:
                merged[-1] = ent
        else:
            merged.append(ent)
    return merged

def post_process_entities(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Romanian-specific post-processing for predicted PII entities."""
    processed: List[Dict[str, Any]] = []
    name_substrings = {}

    for ent in entities:
        s, e, label = ent["start"], ent["end"], ent["label"]
        word = text[s:e].strip()

        # Drop invalid predictions
        if not word or (word != "\n" and word.isspace()):
            continue
        if label == "EMAIL" and "@" not in word:
            continue
        if label == "NUME_PRENUME":
            if not word.istitle() or len(word) == 1 or any(c.isdigit() for c in word) or word in ROMANIAN_NAME_BLACKLIST:
                continue
            name_substrings[word] = True
        if label in {"CNP", "BULETIN", "NUMAR_LICENTA", "PASAPORT"}:
            if any(c in ": -!@#$&%?^+=*<>" for c in word):
                continue
        if label in {"TELEFON_MOBIL", "TELEFON_FIX"}:
            cleaned = re.sub(r"[^\d+]", "", word)
            if not cleaned or len(cleaned) < 7:
                continue
            word = cleaned
            ent["text"] = word
        if label in {"IBAN", "CONT_BANCAR"}:
            ent["text"] = word.replace(" ", "")
        processed.append(ent)

    # Propagate repeated names
    final_entities: List[Dict[str, Any]] = []
    for ent in processed:
        word = text[ent["start"]:ent["end"]].strip()
        if word in name_substrings:
            ent["label"] = "NUME_PRENUME"
        final_entities.append(ent)

    # Merge overlapping spans
    final_entities = merge_overlapping_entities(final_entities)

    return final_entities
