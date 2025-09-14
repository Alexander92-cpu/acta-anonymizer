"""Configuration management utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    base_model: str
    model_max_length: int


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    eval_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    report_to: Optional[str]


@dataclass
class DataConfig:
    """Data configuration."""

    train_file: str
    validation_split: float
    test_split: float
    max_length: int


class ConfigManager:
    """Manage configuration files."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir

    def load_model_config(self) -> ModelConfig:
        """Load model configuration."""
        with open(f"{self.config_dir}/model.yaml", "r") as f:
            config = yaml.safe_load(f)

        return ModelConfig(
            base_model=config["model"]["base_model"],
            model_max_length=config["model"]["model_max_length"],
        )

    def get_lora_config(self, domain: str = "financial") -> Dict[str, Any]:
        """Get LoRA configuration for domain."""
        with open(f"{self.config_dir}/model.yaml", "r") as f:
            config = yaml.safe_load(f)

        return config["domains"][domain]["lora"]

    def load_training_config(
        self,
    ) -> Tuple[TrainingConfig, DataConfig]:
        """Load training configuration."""
        with open(f"{self.config_dir}/training.yaml", "r") as f:
            config = yaml.safe_load(f)

        training = TrainingConfig(
            output_dir=config["training"]["output_dir"],
            num_train_epochs=config["training"]["num_train_epochs"],
            per_device_train_batch_size=config["training"][
                "per_device_train_batch_size"
            ],
            per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            logging_steps=config["training"]["logging_steps"],
            eval_steps=config["training"]["eval_steps"],
            save_steps=config["training"]["save_steps"],
            save_total_limit=config["training"]["save_total_limit"],
            eval_strategy=config["training"]["eval_strategy"],
            save_strategy=config["training"]["save_strategy"],
            load_best_model_at_end=config["training"]["load_best_model_at_end"],
            metric_for_best_model=config["training"]["metric_for_best_model"],
            greater_is_better=config["training"]["greater_is_better"],
            report_to=config["training"]["report_to"],
        )

        data = DataConfig(
            train_file=config["data"]["train_file"],
            validation_split=config["data"]["validation_split"],
            test_split=config["data"]["test_split"],
            max_length=config["data"]["max_length"],
        )

        return training, data

    def load_publishing_config(self, domain: str = "financial") -> Dict[str, Any]:
        """Load publishing configuration."""
        with open(f"{self.config_dir}/publishing.yaml", "r") as f:
            config = yaml.safe_load(f)

        repo_config = config["publishing"]["repository"]
        settings = config["publishing"]["settings"]
        domain_config = config["publishing"]["domains"][domain]

        return {
            "repo_id": f"{repo_config['owner']}/{repo_config['base_name']}{domain_config['repo_suffix']}",
            "private": repo_config["private"],
            "description": domain_config["description"],
            "commit_message": settings["commit_message"],
            "auto_update_metrics": settings["auto_update_metrics"],
            "create_model_card": settings["create_model_card"],
        }

    def load_model_card_config(self, domain: str = "financial") -> Dict[str, Any]:
        """Load model card configuration."""
        with open(f"{self.config_dir}/model_card.yaml", "r") as f:
            config = yaml.safe_load(f)

        card_config = config["model_card"]
        common_tags = card_config["tags"]["common"]
        domain_tags = card_config["tags"].get(domain, [])

        return {
            "metadata": card_config["metadata"],
            "tags": common_tags + domain_tags,
            "training_data_description": card_config["training_data"][domain],
            "usage_example": card_config["usage_examples"][domain]["predict"],
            "key_features": card_config["sections"]["key_features"],
            "use_cases": card_config["sections"]["use_cases"][domain],
        }
