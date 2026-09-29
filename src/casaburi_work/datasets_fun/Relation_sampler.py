from collections import defaultdict
import random
import math
from torch.utils.data import Sampler


class RelationBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.relation_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            relation = dataset[idx]["bias_code"]

            self.relation_to_indices[
                relation
            ].append(idx)

    def __iter__(self):

        all_batches = []

        for relation, indices in \
                self.relation_to_indices.items():

            indices = indices.copy()

            if self.shuffle:
                random.shuffle(indices)

            for start in range(
                0,
                len(indices),
                self.batch_size
            ):
                batch = indices[
                    start:
                    start + self.batch_size
                ]

                if (
                    self.drop_last
                    and len(batch) < self.batch_size
                ):
                    continue

                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):

        total = 0

        for indices in \
                self.relation_to_indices.values():

            if self.drop_last:
                total += (
                    len(indices)
                    // self.batch_size
                )
            else:
                total += math.ceil(
                    len(indices)
                    / self.batch_size
                )

        return total