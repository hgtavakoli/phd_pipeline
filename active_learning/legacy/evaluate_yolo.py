# evaluate_yolo.py

import os
import csv
from ultralytics import YOLO
import config_yolo as config

if __name__ == "__main__":
    results = []

    # Iterate through all rounds
    for i in range(config.AL_ROUNDS + 1):
        model_path = os.path.join(config.OUTPUT_DIR, f"{config.PROJECT_NAME}_round_{i}", "weights", "best.pt")

        if not os.path.exists(model_path):
            print(f"Warning: Model not found for round {i} at {model_path}. Skipping.")
            continue

        print(f"--- Evaluating model for Round {i} ---")
        model = YOLO(model_path)

        # Run validation on the original, full validation set
        metrics = model.val(
            data=config.DATASET_YAML,
            split='val',
            project=config.OUTPUT_DIR,
            name=f"eval_{config.PROJECT_NAME}_round_{i}",
            save_json=True  # To get COCO metrics if needed
        )

        round_results = {
            "round": i,
            "model_path": model_path,
            "mAP50-95(B)": metrics.box.map,
            "mAP50(B)": metrics.box.map50,
            "mAP75(B)": metrics.box.map75,
            "precision(B)": metrics.box.mp,
            "recall(B)": metrics.box.mr
        }
        results.append(round_results)
        print(f"Round {i} mAP50-95: {metrics.box.map:.4f}")

    # Write results to CSV
    output_csv_path = os.path.join(config.OUTPUT_DIR, f"{config.PROJECT_NAME}_evaluation_results.csv")
    with open(output_csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n--- Evaluation complete. Results saved to {output_csv_path} ---")