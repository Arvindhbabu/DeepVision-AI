"""
DeepVision AI

Training History Manager
"""

from pathlib import Path
import json
import csv


class History:
    """
    Stores and manages training history.
    """

    def __init__(self):
        self.records = []

    # --------------------------------------------------
    # Add epoch
    # --------------------------------------------------

    def add(self, **kwargs):
        """
        Add one epoch record.

        Example:
            history.add(
                epoch=1,
                train_loss=0.4,
                val_loss=0.3,
                train_accuracy=0.95,
                val_accuracy=0.94,
            )
        """
        self.records.append(kwargs)

    # --------------------------------------------------
    # Length
    # --------------------------------------------------

    def __len__(self):
        return len(self.records)

    # --------------------------------------------------
    # Best Epoch
    # --------------------------------------------------

    def best(self, metric="val_loss", mode="min"):

        if len(self.records) == 0:
            return None

        if mode == "min":
            return min(
                self.records,
                key=lambda x: x[metric]
            )

        return max(
            self.records,
            key=lambda x: x[metric]
        )

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------

    def save_json(self, path):

        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:

            json.dump(
                self.records,
                f,
                indent=4,
            )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    def save_csv(self, path):

        path = Path(path)

        if len(self.records) == 0:
            return

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.records[0].keys(),
            )

            writer.writeheader()

            writer.writerows(self.records)

    # --------------------------------------------------
    # Return records
    # --------------------------------------------------

    def get(self):

        return self.records