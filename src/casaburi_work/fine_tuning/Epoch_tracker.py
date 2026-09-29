import csv
import json
import os
from collections import defaultdict


class EpochMetricTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, metrics, weight=1):
        """
        metrics: dict
        {
            "bias_loss": tensor/scalar,
            "rank_normal": tensor/scalar,
            ...
        }
        """

        for name, value in metrics.items():

            if hasattr(value, "detach"):
                value = value.detach().item()

            value = float(value)

            self.sums[name] += value * weight
            self.counts[name] += weight

    def compute(self):
        results = {}

        for name in self.sums:
            results[name] = (
                self.sums[name]
                / max(self.counts[name], 1)
            )

        return results

    def save_epoch_metrics(filepath, epoch, metrics):
        row = {
        "epoch": epoch,
        **metrics
        }

        file_exists = os.path.exists(filepath)

        with open(filepath,"a",newline="",encoding="utf-8") as f:

            writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)