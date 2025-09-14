"""Simple training script for financial PII detection."""

from src.acta_anonymizer.trainer import AdapterTrainer


def main() -> None:
    """Train the financial domain adapter."""
    trainer = AdapterTrainer()

    output_dir = trainer.train_from_data_file(
        domain="financial",
        publish_to_hub=True,
        versioning_enabled=True,
    )

    print(f"Training completed. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
