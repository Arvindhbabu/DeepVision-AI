"""
=============================================================
DeepVision AI

Model Evaluation Script

Author : Arvindh Babu
=============================================================
"""

import argparse
import json
from pathlib import Path
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import torch
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="DeepVision AI Evaluation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Evaluation output folder"
    )

    return parser.parse_args()


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("=" * 60)
        print("CUDA AVAILABLE")
        print("=" * 60)
        print("GPU :", torch.cuda.get_device_name(0))
        print("=" * 60)

    else:

        device = torch.device("cpu")

        print("=" * 60)
        print("Running on CPU")
        print("=" * 60)

    return device


# ============================================================
# Load Model
# ============================================================

def load_model(config, checkpoint_path, device):

    print("\nCreating model...")

    model = ModelFactory.create(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("Checkpoint Loaded Successfully")

    return model


# ============================================================
# Prepare Evaluation
# ============================================================

def prepare():

    args = parse_arguments()

    config = load_config(args.config)

    device = get_device()

    print("\nCreating dataloaders...")

    _, _, test_loader = create_dataloaders(config)

    print(f"Test Samples : {len(test_loader.dataset)}")
    print(f"Test Batches : {len(test_loader)}")

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model(
        config,
        args.checkpoint,
        device
    )

    return (
        model,
        test_loader,
        device,
        output_dir,
        config
    )


# ============================================================
# Inference
# ============================================================

def evaluate(model, test_loader, device):

    print("\nRunning inference...")

    all_labels = []
    all_predictions = []
    all_probabilities = []

    all_video_names = []
    all_dataset_names = []

    with torch.no_grad():

        for batch in tqdm(test_loader):

            sequences = batch["sequence"].to(device)

            labels = batch["label"].to(device)

            dataset_names = batch["dataset"]

            video_names = batch["video_name"]

            # Forward Pass
            logits = model(sequences)

            # Softmax probabilities
            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # Predicted class
            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            # Probability of Fake class
            fake_probability = probabilities[:, 1]

            # Store tensors

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                fake_probability.cpu().numpy().tolist()
            )

            # Metadata

            all_video_names.extend(video_names)

            all_dataset_names.extend(dataset_names)

    print("Inference Finished")

    return {

        "labels": all_labels,

        "predictions": all_predictions,

        "probabilities": all_probabilities,

        "video_names": all_video_names,

        "dataset_names": all_dataset_names,

    }


# ============================================================
# Save Prediction CSV
# ============================================================

def save_predictions(results, output_dir):

    print("Saving predictions...")

    dataframe = pd.DataFrame({

        "video_name": results["video_names"],

        "dataset": results["dataset_names"],

        "true_label": results["labels"],

        "predicted_label": results["predictions"],

        "fake_probability": results["probabilities"]

    })

    dataframe["correct"] = (

        dataframe["true_label"]

        ==

        dataframe["predicted_label"]

    )

    csv_path = output_dir / "predictions.csv"

    dataframe.to_csv(

        csv_path,

        index=False

    )

    print(f"Saved : {csv_path}")

    return dataframe


# ============================================================
# Metrics
# ============================================================

def compute_metrics(results, output_dir):

    print("\nComputing evaluation metrics...")

    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Real", "Fake"],
        columns=["Pred Real", "Pred Fake"]
    )

    cm_df.to_csv(
        output_dir / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        target_names=["Real", "Fake"],
        digits=4,
        zero_division=0
    )

    with open(
        output_dir / "classification_report.txt",
        "w"
    ) as file:

        file.write(report)

    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "num_samples": len(labels)

    }

    with open(
        output_dir / "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print("\n==============================")
    print("Evaluation Results")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    return metrics, cm

# ============================================================
# Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(cm, output_dir):

    print("Generating Confusion Matrix...")

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"]
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = output_dir / "confusion_matrix.png"

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

    print(f"Saved : {save_path}")


# ============================================================
# Plot ROC Curve
# ============================================================

def plot_roc(results, output_dir):

    print("Generating ROC Curve...")

    labels = results["labels"]
    probabilities = results["probabilities"]

    fpr, tpr, _ = roc_curve(
        labels,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()

    save_path = output_dir / "roc_curve.png"

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

    print(f"Saved : {save_path}")


# ============================================================
# Main
# ============================================================

def main():

    (
        model,
        test_loader,
        device,
        output_dir,
        config
    ) = prepare()

    results = evaluate(
        model,
        test_loader,
        device
    )

    save_predictions(
        results,
        output_dir
    )

    metrics, cm = compute_metrics(
        results,
        output_dir
    )

    plot_confusion_matrix(
        cm,
        output_dir
    )

    plot_roc(
        results,
        output_dir
    )

    print("\n" + "=" * 60)
    print("Evaluation Completed Successfully")
    print("=" * 60)

    print(f"Results saved to : {output_dir}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()

