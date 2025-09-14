"""Data loading utilities for FinTech PII dataset."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from .config import ConfigManager


class DataLoader:
    """Load and preprocess FinTech PII dataset."""

    def __init__(
        self,
        data_path: str,
        config_dir: str = "configs",
    ) -> None:
        self.data_path: str = data_path
        self.config_manager: ConfigManager = ConfigManager(config_dir)

        model_config = self.config_manager.load_model_config()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)

        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict[int, str]] = None
        self._max_length: int = model_config.model_max_length

    def load_data(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Load data from JSON file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
        return data

    def _tokenize_and_align_labels_with_ids(
        self, tokens: List[str], label_ids: List[int]
    ) -> Dict[str, List[Any]]:
        """Tokenize and align labels when label IDs are already provided."""
        # Tokenize with the model's tokenizer
        tokenized = self.tokenizer(
            tokens,
            truncation=True,
            max_length=self._max_length,
            padding=False,
            return_offsets_mapping=True,
            is_split_into_words=True,
            return_overflowing_tokens=False,
        )

        word_ids = (
            tokenized.word_ids()
        )  # list mapping each sub-token -> word index or None
        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # special token or padding
            else:
                # here we label all subword tokens with the same word label
                aligned_labels.append(label_ids[word_idx])

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": aligned_labels,
        }

    def _build_label_mapping(self, data: List[Dict[str, Any]]) -> None:
        """Build label to ID mapping from training data using ner_ids."""
        label_to_id = {}
        id_to_label = {}

        for item in data:
            if "ner_tags" in item and "ner_ids" in item:
                # Use the provided ner_ids mapping
                for tag, tag_id in zip(item["ner_tags"], item["ner_ids"]):
                    if tag not in label_to_id:
                        label_to_id[tag] = tag_id
                        id_to_label[tag_id] = tag

        self._label_to_id = label_to_id
        self._id_to_label = id_to_label

    def preprocess_data(self, data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Preprocess data into training format."""
        if self._label_to_id is None:
            self._build_label_mapping(data)

        processed_data: Dict[str, List[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for item in data:
            tokens = item["tokens"]
            label_ids = item["ner_ids"]
            tokenized_data = self._tokenize_and_align_labels_with_ids(tokens, label_ids)

            processed_data["input_ids"].append(tokenized_data["input_ids"])
            processed_data["attention_mask"].append(tokenized_data["attention_mask"])
            processed_data["labels"].append(tokenized_data["labels"])

        return processed_data

    def load_and_split(
        self,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Load data and split into train/val/test sets."""

        # Load raw data
        raw_data: Union[Dict[str, Any], List[Dict[str, Any]]] = self.load_data()

        # Handle different data formats
        if isinstance(raw_data, list):
            data_list = raw_data
        elif isinstance(raw_data, dict) and "data" in raw_data:
            data_list = list(raw_data["data"])
        else:
            raise ValueError("Unsupported data format")

        # Build label mapping if not already done
        if self._label_to_id is None:
            self._build_label_mapping(data_list)

        # Preprocess data
        processed_data: Dict[str, List[Any]] = self.preprocess_data(data_list)

        # Split data
        combined_data: List[Tuple[List[int], List[int], List[int]]] = list(
            zip(
                processed_data["input_ids"],
                processed_data["attention_mask"],
                processed_data["labels"],
            )
        )

        train_data, temp_data = train_test_split(
            combined_data,
            test_size=validation_split + test_split,
            random_state=random_seed,
        )

        val_size: float = validation_split / (validation_split + test_split)
        val_data, test_data = train_test_split(
            temp_data, test_size=1 - val_size, random_state=random_seed
        )

        # Convert back to dict format
        def to_dataset_dict(
            data_tuples: List[Tuple[List[int], List[int], List[int]]],
        ) -> Dict[str, List[Any]]:
            if not data_tuples:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            input_ids, attention_mask, labels = zip(*data_tuples)
            return {
                "input_ids": list(input_ids),
                "attention_mask": list(attention_mask),
                "labels": list(labels),
            }

        train_dict: Dict[str, List[Any]] = to_dataset_dict(train_data)
        val_dict: Dict[str, List[Any]] = to_dataset_dict(val_data)
        test_dict: Dict[str, List[Any]] = to_dataset_dict(test_data)

        # Create datasets
        train_dataset: Dataset = Dataset.from_dict(train_dict)
        val_dataset: Dataset = Dataset.from_dict(val_dict)
        test_dataset: Dataset = Dataset.from_dict(test_dict)

        return train_dataset, val_dataset, test_dataset

    @property
    def label_to_id(self) -> Dict[str, int]:
        """Get label to ID mapping."""
        return self._label_to_id.copy() if self._label_to_id else {}

    @property
    def id_to_label(self) -> Dict[int, str]:
        """Get ID to label mapping."""
        return self._id_to_label.copy() if self._id_to_label else {}

    @property
    def num_labels(self) -> int:
        """Get number of labels (max ID + 1 for 0-based indexing)."""
        return max(self._id_to_label.keys()) + 1 if self._id_to_label else 0

    def get_data_collator(self) -> DataCollatorForTokenClassification:
        """Get data collator for training."""
        return DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self._max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
