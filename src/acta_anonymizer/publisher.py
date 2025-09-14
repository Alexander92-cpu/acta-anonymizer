"""HuggingFace Hub publishing utilities."""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import yaml
from huggingface_hub import HfApi, create_repo


class ModelPublisher:
    """Publish models to HuggingFace Hub."""

    def __init__(self):
        self.api = HfApi()
        self.token = os.getenv("HF_TOKEN")

    def publish_model(
        self,
        model_path: str,
        publishing_config: Dict[str, Any],
        model_card_config: Dict[str, Any],
        metrics: Dict[str, Any] = None,
        version: Optional[str] = None,
        versioning_enabled: bool = True,
    ) -> str:
        """Publish model to HuggingFace Hub with optional versioning."""

        if not self.token:
            raise ValueError("HF_TOKEN environment variable not set")

        repo_id = publishing_config["repo_id"]
        private = publishing_config["private"]
        commit_message = publishing_config["commit_message"]

        # Generate version if not provided
        if versioning_enabled and not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create repository
        try:
            create_repo(
                repo_id=repo_id, token=self.token, private=private, exist_ok=True
            )
        except Exception as e:
            print(f"Repository creation failed or already exists: {e}")

        if versioning_enabled and version:
            # Upload to versioned folder
            print(f"Uploading model to versions/{version}/")
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                path_in_repo=f"versions/{version}/",
                token=self.token,
                commit_message=f"{commit_message} - Version {version}",
            )

            # Upload to root (latest version)
            print("Updating latest version at root level")
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                path_in_repo="",
                token=self.token,
                commit_message=f"{commit_message} - Latest (v{version})",
            )

            # Update version index
            self._update_version_index(repo_id, version, metrics)
        else:
            # Original behavior - upload directly to root
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                token=self.token,
                commit_message=commit_message,
            )

        # Create model card if requested
        if publishing_config.get("create_model_card", True):
            self._create_model_card(
                repo_id,
                model_card_config,
                publishing_config["description"],
                version if versioning_enabled else None,
            )

        # Upload metrics if available and requested
        if metrics and publishing_config.get("auto_update_metrics", True):
            self.update_metrics(
                repo_id, metrics, version if versioning_enabled else None
            )

        return f"https://huggingface.co/{repo_id}"

    def _create_model_card(
        self,
        repo_id: str,
        model_card_config: Dict[str, Any],
        description: str,
        version: Optional[str] = None,
    ):
        """Create model card for the repository using config."""
        metadata = model_card_config["metadata"]
        tags = model_card_config["tags"]

        # Build YAML frontmatter
        yaml_tags = "\n".join([f"- {tag}" for tag in tags])
        yaml_languages = "\n".join([f"- {lang}" for lang in metadata["language"]])

        version_info = f"\n\n**Current Version**: {version}" if version else ""

        model_card = f"""---
license: {metadata["license"]}
language:
{yaml_languages}
base_model: {metadata["base_model"]}
tags:
{yaml_tags}
---

# {repo_id}

{description}{version_info}

## Key Features

{chr(10).join([f"- {feature}" for feature in model_card_config["key_features"]])}

## Use Cases

{chr(10).join([f"- {use_case}" for use_case in model_card_config["use_cases"]])}

## Training Data

{model_card_config["training_data_description"]}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load base model
model = AutoModelForTokenClassification.from_pretrained("{metadata["base_model"]}")
tokenizer = AutoTokenizer.from_pretrained("{metadata["base_model"]}")

# Load adapter
model = PeftModel.from_pretrained(model, "{repo_id}")

# Create pipeline
ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# Example usage
text = "{model_card_config["usage_example"]}"
entities = ner_pipeline(text)
print(entities)
```

## Training

This model was trained using LoRA (Low-Rank Adaptation) on synthetic Moldovan PII data.

## Versions

- **Latest**: Root level contains the most recent version
- **Archived**: Previous versions are stored in `versions/` folder
- **Version Index**: See `version_history.yaml` for complete version history
"""

        # Upload model card
        self.api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=self.token,
            commit_message="Add model card",
        )

    def update_metrics(
        self, repo_id: str, metrics: Dict[str, Any], version: Optional[str] = None
    ):
        """Update model metrics in the repository."""
        try:
            metrics_content = yaml.dump(metrics, default_flow_style=False)

            # Upload main metrics file
            self.api.upload_file(
                path_or_fileobj=metrics_content.encode(),
                path_in_repo="metrics.yaml",
                repo_id=repo_id,
                token=self.token,
                commit_message="Add evaluation metrics",
            )

            # Upload versioned metrics if version is provided
            if version:
                self.api.upload_file(
                    path_or_fileobj=metrics_content.encode(),
                    path_in_repo=f"versions/{version}/metrics.yaml",
                    repo_id=repo_id,
                    token=self.token,
                    commit_message=f"Add metrics for version {version}",
                )
        except Exception as e:
            print(f"Failed to upload metrics: {e}")

    def _update_version_index(
        self, repo_id: str, version: str, metrics: Optional[Dict[str, Any]] = None
    ):
        """Update version history index."""
        try:
            # Try to download existing version history
            try:
                history_content = self.api.hf_hub_download(
                    repo_id=repo_id,
                    filename="version_history.yaml",
                    token=self.token,
                )
                with open(history_content, "r") as f:
                    version_history = yaml.safe_load(f) or {"versions": []}
            except:
                # Create new history if file doesn't exist
                version_history = {"versions": []}

            # Add new version entry
            version_entry = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "path": f"versions/{version}/",
            }

            if metrics:
                # Add key metrics to version entry
                version_entry["metrics"] = {
                    k: v
                    for k, v in metrics.items()
                    if k.startswith("eval_") and isinstance(v, (int, float))
                }

            # Add to history (latest first)
            version_history["versions"].insert(0, version_entry)

            # Keep only last 20 versions in history
            version_history["versions"] = version_history["versions"][:20]

            # Upload updated history
            history_content = yaml.dump(version_history, default_flow_style=False)
            self.api.upload_file(
                path_or_fileobj=history_content.encode(),
                path_in_repo="version_history.yaml",
                repo_id=repo_id,
                token=self.token,
                commit_message=f"Update version history - add {version}",
            )
        except Exception as e:
            print(f"Failed to update version history: {e}")
