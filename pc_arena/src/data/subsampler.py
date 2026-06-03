import torch
import math
from torch.utils.data import Sampler

class DistributedSubsetSampler(Sampler):
    """
    A Distributed Sampler that subsamples a dataset at each epoch.
    """
    def __init__(self, dataset, subset_size, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        
        self.dataset = dataset
        self.subset_size = subset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        
        self.effective_subset_size = min(self.subset_size, len(self.dataset))
        self.num_samples = int(math.ceil(self.effective_subset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        subset_indices = indices[:self.effective_subset_size]
        
        padding_size = self.total_size - len(subset_indices)
        if padding_size > 0:
            subset_indices += subset_indices[:padding_size]
        assert len(subset_indices) == self.total_size

        indices_for_rank = subset_indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices_for_rank) == self.num_samples

        return iter(indices_for_rank)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. This ensures a different random subset and
        shuffle order for each epoch.
        """
        self.epoch = epoch