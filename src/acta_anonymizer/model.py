"""Clean, production-ready anonymization model."""

from typing import Any, Dict, List, Optional, Union

from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import Pipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import ConfigManager, ModelConfig


class AnonymizerModel:
    """Text anonymizer using BERT-based NER with optional PEFT adapters."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        config_dir: str = "configs",
    ) -> None:
        self.config_manager: ConfigManager = ConfigManager(config_dir)
        self.model_config: ModelConfig = self.config_manager.load_model_config()

        self.model_name: str = model_name or self.model_config.base_model
        self.adapter_path: Optional[str] = adapter_path
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model: Optional[Union[PreTrainedModel, PeftModel]] = None
        self._pipeline: Optional[Pipeline] = None

    def load(self) -> "AnonymizerModel":
        """Load the model and tokenizer."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        if self.adapter_path:
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

        self._pipeline = pipeline(
            "token-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            aggregation_strategy="simple",
        )
        return self

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self._pipeline:
            raise RuntimeError("Model not loaded. Call load() first.")

        entities: List[Dict[str, Any]] = self._pipeline(text)
        return [
            {
                "text": str(entity["word"]),
                "label": str(entity["entity_group"]),
                "confidence": float(entity["score"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
            }
            for entity in entities
        ]
