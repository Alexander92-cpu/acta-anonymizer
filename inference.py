"""Simple inference script for Romanian NER model."""

from typing import Any, Dict, List

import yaml
from peft import PeftModel
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)


class RomanianNERInference:
    """Inference class for Romanian Named Entity Recognition."""

    def __init__(self, config_path: str = "configs/inference.yaml") -> None:
        """Initialize the model and tokenizer."""
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.config = config["inference"]

        # Load models
        base_model_name = self.config["base_model"]
        adapter_model_name = self.config["adapter_model"]

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        try:
            # Try to load with PEFT adapter
            model = AutoModelForTokenClassification.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(model, adapter_model_name)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    "Size mismatch detected. Loading base model with correct config..."
                )
                # Load base model but modify config for correct number of labels
                model = AutoModelForTokenClassification.from_pretrained(base_model_name)

                # Get the saved classifier state from the adapter
                import os

                from huggingface_hub import hf_hub_download
                from safetensors import safe_open

                # Check if adapter_model_name is a local path or HuggingFace model
                if os.path.exists(adapter_model_name):
                    # Local path
                    adapter_model_path = os.path.join(
                        adapter_model_name, "adapter_model.safetensors"
                    )
                    print(f"Loading from local path: {adapter_model_path}")
                else:
                    # HuggingFace Hub - download the safetensors file
                    try:
                        adapter_model_path = hf_hub_download(
                            repo_id=adapter_model_name,
                            filename="adapter_model.safetensors",
                            cache_dir="./models/cache",
                        )
                        print(f"Downloaded from HuggingFace Hub: {adapter_model_name}")
                    except Exception as hf_error:
                        print(f"Failed to download from HuggingFace Hub: {hf_error}")
                        # Fallback to local path if exists
                        local_fallback = "./models/financial_adapter_*"
                        import glob

                        local_models = glob.glob(local_fallback)
                        if local_models:
                            fallback_path = max(local_models)  # Get the latest one
                            adapter_model_path = os.path.join(
                                fallback_path, "adapter_model.safetensors"
                            )
                            print(f"Using local fallback: {fallback_path}")
                        else:
                            raise RuntimeError(
                                f"Cannot find adapter model locally or on HuggingFace Hub: {adapter_model_name}"
                            )

                with safe_open(adapter_model_path, framework="pt", device="cpu") as f:
                    # Find classifier weights to determine number of labels
                    classifier_key = None
                    for key in f.keys():
                        if "classifier" in key and "weight" in key:
                            classifier_key = key
                            break

                    if classifier_key:
                        classifier_weight = f.get_tensor(classifier_key)
                        num_labels = classifier_weight.shape[0]

                        # Update model config and resize classifier
                        model.config.num_labels = num_labels
                        import torch.nn as nn

                        model.classifier = nn.Linear(
                            model.classifier.in_features, num_labels
                        )

                        # Load the saved classifier weights
                        classifier_bias_key = classifier_key.replace("weight", "bias")
                        if classifier_bias_key in f.keys():
                            classifier_bias = f.get_tensor(classifier_bias_key)
                            model.classifier.weight.data = classifier_weight
                            model.classifier.bias.data = classifier_bias

                        print(
                            f"✓ Loaded model with {num_labels} labels using saved classifier weights"
                        )
                    else:
                        raise RuntimeError(
                            "Could not find classifier weights in adapter model"
                        )
            else:
                raise e

        self.pipe: Pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self.config["aggregation_strategy"],
        )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Run NER prediction on text."""
        predictions = self.pipe(text)
        return predictions

    def format_predictions(
        self, text: str, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format predictions for better readability."""
        print(f"\nText: {text}")
        print("Entities found:")
        for pred in predictions:
            entity_word = pred["word"]
            entity_group = pred["entity_group"]
            confidence = pred["score"]
            print(f"  - {entity_word}: {entity_group} (confidence: {confidence:.3f})")
        return predictions


def main() -> None:
    """Test the Romanian NER model with examples."""
    # Load config to get test examples
    with open("configs/inference.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ner = RomanianNERInference()
    test_examples: List[str] = config["test_examples"]

    print("=== Romanian NER Inference Test ===\n")

    for i, text in enumerate(test_examples, 1):
        print(f"Example {i}:")
        predictions = ner.predict(text)
        ner.format_predictions(text, predictions)
        print("-" * 80)


if __name__ == "__main__":
    main()
