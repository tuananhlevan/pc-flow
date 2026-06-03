import os
import math
from typing import Tuple, Dict, Any, Optional
import argparse
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import pyjuice as juice

from src.utils import instantiate_from_config, collect_data_from_dsets
from src.layers.monarchlayer import create_monarch_layers
from src.sgd import SGDWrapper
from training.utils import find_largest_epoch

def build_or_load_pc(rank: int, args: argparse.Namespace, paths: Dict[str, str], dsets: Any) -> Tuple[Any, int]:
    """Constructs a new Probabilistic Circuit or loads a checkpoint across DDP ranks."""
    epoch_start = 1
    root_ns = None

    if rank == 0:
        if args.resume:
            assert os.path.exists(paths["pcfile_last"]) and os.path.exists(paths["logfile"])
            root_ns = juice.load(paths["pcfile_last"])
            epoch_start = (find_largest_epoch(paths["logfile"]) or 0) + 1
            print(f"[rank 0] PC resumed from epoch {epoch_start}...")
        else:
            model_config = OmegaConf.load(f"./configs/model/{args.model_config}.yaml")
            model_kwargs = {}
            
            for k, v in list(model_config["params"].items()):
                if isinstance(v, str) and v.startswith("__train_data__:"):
                    num_samples = int(v.split(":")[1])
                    data = collect_data_from_dsets(dsets, num_samples=num_samples, split="train")
                    model_config["params"].pop(k)
                    model_kwargs[k] = data.cuda()
            
            if "monarch" in args.model_config:
                if 'num_latents' in model_config['params']:
                    model_config['params']['block_size'] = int(model_config['params']['num_latents'] ** 0.5)
                model_config['params']['homogeneous_inputs'] = True
                model_kwargs['layer_fn'] = create_monarch_layers
            
            print("[rank 0] Constructing PC...")
            root_ns = instantiate_from_config(model_config, recursive=True, **model_kwargs)
            juice.save(paths["pcfile_last"], root_ns)
            print("[rank 0] PC constructed and saved...")

    dist.barrier()
    if rank != 0:
        root_ns = juice.load(paths["pcfile_last"])
        if args.resume:
            epoch_start = (find_largest_epoch(paths["logfile"]) or 0) + 1
        print(f"[rank {rank}] PC safely loaded from rank 0 checkpoint...")
    dist.barrier()

    return root_ns, epoch_start


def configure_optimizer(optim_config: Any, world_size: int, gpu_batch_size: int, loader_len: int) -> Dict[str, Any]:
    """Resolves structural hyperparameters based on the preferred optimization mode."""
    mode = optim_config["mode"]
    opt_meta = {"mode": mode, "num_epochs": optim_config["num_epochs"], "momentum": 0.0, "step_size": 1.0, "adam_kwargs": {}}

    if mode == "full_em":
        opt_meta["niters_per_update"] = loader_len
    elif mode in ["mini_em", "mini_em_scaled", "adam"]:
        opt_meta["step_size"] = optim_config["lr"] if mode == "adam" else optim_config["step_size"]
        opt_meta["niters_per_update"] = optim_config["batch_size"] // world_size // gpu_batch_size
        assert opt_meta["niters_per_update"] > 0, "Batch scaling constraint violated. Check batch sizes vs world size."
        
        if mode == "adam":
            opt_meta["cum_batch_size"] = optim_config["batch_size"]
            if "beta1" in optim_config: opt_meta["adam_kwargs"]["beta1"] = optim_config["beta1"]
            if "beta2" in optim_config: opt_meta["adam_kwargs"]["beta2"] = optim_config["beta2"]
        elif "momentum" in optim_config:
            opt_meta["momentum"] = optim_config["momentum"]
    else:
        raise NotImplementedError(f"Optim mode {mode} not found.")
        
    return opt_meta


def run_evaluation(rank: int, world_size: int, pc: Any, pcopt: Optional[SGDWrapper], vl_loader: Any, device: torch.device, optim_mode: str) -> float:
    """Evaluates the network on validation datasets and computes global reduced loss."""
    local_ll_sum = 0.0
    for x in vl_loader:
        x = x.to(device)
        with torch.cuda.device(device):
            lls = pc(x, propagation_alg="LL")
            if optim_mode == "adam" and pcopt is not None:
                pcopt.partition_eval(negate_pflows=True)
                lls = lls - pc.node_mars[-1, 0]
        local_ll_sum += lls.mean().item()

    stats = torch.tensor([local_ll_sum], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats[0].item() / world_size / len(vl_loader)