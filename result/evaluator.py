from initial.evaluator import Evaluator, load_dataset
from gliner_run import AnonymizerGLiNER
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
model.save_pretrained("gliner")

# examples = load_dataset("data/mock_subset_200.json", limit=200)
examples = load_dataset("data/synthetic_moldova_pii_data.json", limit=1000)

client = AnonymizerGLiNER(model_path="gliner")


evaluator = Evaluator(client, ignore_labels=True)
metrics = evaluator.evaluate(examples)

print("Evaluation results:")
for k, v in metrics.items():
    print(f"{k}: {v}")
