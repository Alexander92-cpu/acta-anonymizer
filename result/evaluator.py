import argparse
from initial.evaluator import Evaluator, load_dataset
from gliner_run import AnonymizerGLiNER
from gliner import GLiNER

def main():
    parser = argparse.ArgumentParser(description="Evaluate GLiNER model on a dataset")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_moldova_pii_data.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of examples to load from dataset"
    )
    args = parser.parse_args()

    # Load model
    model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    model.save_pretrained("gliner")

    # Load dataset
    examples = load_dataset(args.dataset, limit=args.limit)

    # Initialize client
    client = AnonymizerGLiNER(model_path="gliner", device=args.device)

    # Evaluate
    evaluator = Evaluator(client, ignore_labels=True)
    metrics = evaluator.evaluate(examples)

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
