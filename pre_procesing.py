import json


def read_bio_json(filepath):
    """
    Reads a JSON file with BIO-tagged sentences.
    Expects each entry to have 'tokens' and 'tags'.
    Returns a list of sentences, where each sentence is a list of (token, label) pairs.
    """
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        tokens = entry.get("tokens", [])
        tags = entry.get("ner_tags", [])

        if len(tokens) != len(tags):
            raise ValueError(f"Mismatch between tokens and tags in entry: {entry}")

        sentence = list(zip(tokens, tags))
        sentences.append(sentence)

    return sentences

def read_clean_str(string):
    return " ".join([s[0] for s in string]), " ".join([s[1] for s in string])

# Example usage:
if __name__ == "__main__":
    filepath = "data/synthetic_moldova_pii_data.json"  # Replace with your BIO file
    sentences = read_bio_json(filepath)

    # Print the first 2 sentences
    for sent in sentences[:2]:
        # a, b = read_clean_str(sent)
        # print(a)
        # print(b)
        print(sent)
