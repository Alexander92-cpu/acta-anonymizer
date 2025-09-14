"""Clean, production-ready adapter training."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy.typing as npt
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import ConfigManager, ModelConfig
from .data_loader import DataLoader
from .evaluator import Evaluator
from .publisher import ModelPublisher


class AdapterTrainer:
    """Train PEFT adapters for domain-specific anonymization."""

    def __init__(self, config_dir: str = "configs") -> None:
        self.config_manager: ConfigManager = ConfigManager(config_dir)
        self.model_config: ModelConfig = self.config_manager.load_model_config()
        self.base_model: str = self.model_config.base_model
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model: Optional[PreTrainedModel] = None
        self._evaluator: Optional[Evaluator] = None

    def _load_base_model(self) -> None:
        """Load base model and tokenizer."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._model = AutoModelForTokenClassification.from_pretrained(self.base_model)

    def _create_lora_config(self, domain: str = "financial") -> LoraConfig:
        """Create LoRA configuration from config file."""
        lora_config: Dict[str, Any] = self.config_manager.get_lora_config(domain)

        config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=int(lora_config["r"]),
            inference_mode=False,
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            target_modules=list(lora_config["target_modules"]),
        )

        return config

    def _compute_metrics(
        self, eval_pred: Tuple[npt.NDArray[Any], npt.NDArray[Any]]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        print("DEBUG: _compute_metrics called")
        if self._evaluator is None:
            id2label = self.peft_model.config.id2label or {}
            self._evaluator = Evaluator(id2label)
        return self._evaluator.evaluate(eval_pred)

    def train_from_data_file(
        self,
        domain: str = "financial",
        publish_to_hub: bool = False,
        version: Optional[str] = None,
        versioning_enabled: bool = True,
    ) -> str:
        """Train adapter from data file and return output path."""

        # Load configurations
        training_config, data_config = self.config_manager.load_training_config()

        # Load and split data
        data_loader: DataLoader = DataLoader(data_config.train_file)
        train_dataset, val_dataset, test_dataset = data_loader.load_and_split(
            validation_split=data_config.validation_split,
            test_split=data_config.test_split,
        )

        # Load base model with correct number of labels
        if not self._model:
            self._load_base_model()
            # Resize model to match custom labels
            if self._model and data_loader.num_labels > 0:
                # Update config first
                self._model.config.num_labels = data_loader.num_labels
                self._model.config.id2label = data_loader.id_to_label
                self._model.config.label2id = data_loader.label_to_id

                # Safely recreate classifier layer with correct size
                old_classifier = self._model.classifier
                device = old_classifier.weight.device
                dtype = old_classifier.weight.dtype

                new_classifier = nn.Linear(
                    old_classifier.in_features,
                    data_loader.num_labels,
                    device=device,
                    dtype=dtype,
                )

                # Initialize with small random weights
                nn.init.normal_(new_classifier.weight, std=0.02)
                nn.init.zeros_(new_classifier.bias)

                self._model.classifier = new_classifier
                self._model.num_labels = data_loader.num_labels

        if self._model is None:
            raise RuntimeError("Failed to load base model")

        # Setup PEFT
        lora_config: LoraConfig = self._create_lora_config(domain)
        self.peft_model: PeftModel = get_peft_model(self._model, lora_config)

        # Verify classifier layer is trainable
        classifier_trainable = False
        for name, param in self.peft_model.named_parameters():
            if "classifier" in name and param.requires_grad:
                classifier_trainable = True
                print(f"✓ Classifier parameter '{name}' is trainable")

        if not classifier_trainable:
            raise RuntimeError(
                "Classifier layer is not trainable! Check modules_to_save configuration."
            )

        print(
            f"✓ PEFT model created with {self.peft_model.num_parameters()} total parameters"
        )
        print(
            f"✓ Trainable parameters: {self.peft_model.num_parameters(only_trainable=True)}"
        )
        print(f"✓ Classifier layer with {data_loader.num_labels} labels is trainable")

        # Generate unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if version:
            # Use provided version in directory name
            unique_suffix = version.replace(".", "_").replace("/", "_")
            output_dir = (
                f"{training_config.output_dir}/{domain}_adapter_{unique_suffix}"
            )
        else:
            # Use timestamp for unique directory
            output_dir = f"{training_config.output_dir}/{domain}_adapter_{timestamp}"

        # Training arguments from config
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            num_train_epochs=training_config.num_train_epochs,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            eval_steps=training_config.eval_steps,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            eval_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            report_to=training_config.report_to,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(self._tokenizer)

        # Setup evaluator
        self._evaluator = Evaluator(data_loader.id_to_label)

        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self._tokenizer,
            compute_metrics=self._compute_metrics,
        )

        # Train
        trainer.train()
        self.peft_model.save_pretrained(output_dir)

        config_dict = self.peft_model.config.to_dict()

        # Ensure label mappings are included
        config_dict.update({
            "num_labels": data_loader.num_labels,
            "id2label": data_loader.id_to_label,
            "label2id": data_loader.label_to_id,
        })

        config_path = Path(output_dir) / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved complete config with {len(data_loader.id_to_label)} labels")

        test_results = trainer.evaluate(test_dataset)
        print("Test Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

        # Publish to HuggingFace Hub if requested
        if publish_to_hub:
            try:
                # Load publishing and model card configs
                publishing_config = self.config_manager.load_publishing_config(domain)
                model_card_config = self.config_manager.load_model_card_config(domain)

                publisher = ModelPublisher()
                repo_url = publisher.publish_model(
                    model_path=output_dir,
                    publishing_config=publishing_config,
                    model_card_config=model_card_config,
                    metrics=test_results,
                    version=version,
                    versioning_enabled=versioning_enabled,
                )
                print(f"Model published to: {repo_url}")
            except Exception as e:
                print(f"Publishing failed: {e}")
                print("Training completed but model was not published.")

        return output_dir
