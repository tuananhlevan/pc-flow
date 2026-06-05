import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import argparse
import os
import torch.distributed as dist
import pyjuice as juice
import time
from omegaconf import OmegaConf
import sys

sys.path.append("./")

from src.utils import instantiate_from_config, ProgressBar
from src.data.subsampler import DistributedSubsetSampler
from src.sgd import SGDWrapper
from training.utils import ddp_setup, get_free_port, mkdir_p, copy_configs, resolve_tuple
from training.engine import build_or_load_pc

sys.path.append("../")
from rec_flow.rf import RF
from rec_flow.dit import DiT_Llama
import wandb

# ---------------------------------------------------------
# THE TEMPORAL INJECTOR (Hybrid Neural-PC)
# ---------------------------------------------------------
class TimeMLP(nn.Module):
    def __init__(self, num_leaves):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, num_leaves * 2) 
        )
        
        # NEW: Force the initial outputs to be exactly zero.
        # This guarantees mu=0 and log_var=0 (sigma=1) on the first forward pass.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, t):
        out = self.net(t)
        mu, log_var = out.chunk(2, dim=-1)
        sigma = torch.exp(0.5 * log_var) + 1e-2 
        return mu, sigma

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Flow-PC with Velocity Matching")
    parser.add_argument("--st-data-config", type = str, default = "imagenet32")
    parser.add_argument("--data-config", type = str, default = "imagenet32")
    parser.add_argument("--model-config", type = str, default = "hclt_256")
    parser.add_argument("--gpu-batch-size", type = int, default = 256)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--num-epochs", type = int, default = 100)
    parser.add_argument("--port", type = int, default = 0)
    parser.add_argument("--resume", default = False, action = "store_true")
    
    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pc-arena")
    return parser.parse_args()


def main(rank, world_size, args):
    torch.set_num_threads(8)
    OmegaConf.register_new_resolver('as_tuple', resolve_tuple)
    ddp_setup(rank, world_size, args.port)
    device = torch.device(f'cuda:{rank}')

    # Logging Setup
    base_folder = os.path.join("logs/", f"{args.data_config}/[{args.model_config}]-[flow-matching]/")
    # base_folder = os.path.join("logs/", f"{args.data_config}/[{args.model_config}]-[flow-matching-uncon]/")
    mkdir_p(base_folder)
    logfile = os.path.join(base_folder, "logs.txt")
    pcfile_last = os.path.join(base_folder, "last.jpc")

    # Structure-Dataset Setup
    st_data_config = OmegaConf.load(os.path.join("./configs/data/", args.st_data_config + ".yaml"))
    if args.gpu_batch_size > 0:
        st_data_config["params"]["batch_size"] = args.gpu_batch_size
    st_dsets = instantiate_from_config(st_data_config)
    st_dsets.setup()

    # Training-Dataset Setup
    data_config = OmegaConf.load(os.path.join("./configs/data/", args.data_config + ".yaml"))
    if args.gpu_batch_size > 0:
        data_config["params"]["batch_size"] = args.gpu_batch_size
    dsets = instantiate_from_config(data_config)
    dsets.setup()

    # Build or Load the Static Spatial PC
    paths = {"pcfile_last": pcfile_last, "logfile": logfile}
    root_ns, epoch_start = build_or_load_pc(rank, args, paths, st_dsets)
    
    pc = juice.compile(root_ns)
    pc.to(device)

    # Extract the number of continuous leaf nodes to scale the MLP
    num_leaves = pc.input_layer_group[0].params.size(0) // 2 
    
    mlp = TimeMLP(num_leaves=num_leaves).to(device)
    optimizer = torch.optim.Adam([
        {'params': mlp.parameters(), 'lr': 1e-3},
        {'params': pc.parameters(), 'lr': 1e-2}
    ])
    
    if args.resume:
        sd = torch.load(os.path.join(base_folder, "last_mlp.pt"), map_location=device)
        mlp.load_state_dict(sd)
        optim_state = torch.load(os.path.join(base_folder, "last_optim.pt"), map_location=device)
        optimizer.load_state_dict(optim_state)
    
    # Wrap MLP in DDP
    mlp = torch.nn.parallel.DistributedDataParallel(mlp, device_ids=[rank])
    
    channels = 1
    rf_dit = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()
    rf_dit.load_state_dict(torch.load("../contents/epoch_100/weight.pt"))
    
    rf_dit.eval()

    train_sampler = DistributedSubsetSampler(dsets.datasets["train"], subset_size=1000000, shuffle=True)
    tr_loader = dsets._train_dataloader(sampler = train_sampler)

    if rank == 0:
        progress_bar = ProgressBar(args.num_epochs, len(tr_loader), ["Flow_MSE"], cumulate_statistics = True)
        if args.wandb:
            wandb.init(project=args.wandb_project, name=f"FlowPC-{args.data_config}")

    step_count = 0

    # ---------------------------------------------------------
    # CONTINUOUS FLOW MATCHING LOOP
    # ---------------------------------------------------------
    for epoch in range(1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        if rank == 0: progress_bar.new_epoch_begin()

        for batch_data in tr_loader:
            if isinstance(batch_data, torch.Tensor):
                batch = batch_data.to(device)
            else:
                batch = batch_data[0].to(device)
                
            b = batch.size(0)
            batch = batch.view(b, -1, 1026)
            num_anchors = batch.size(1)
            
            optimizer.zero_grad()
            
            num_pairs = batch.size(1)
            time_idx = torch.randint(0, num_pairs, (1,)).item()
            
            # for time_idx in range(num_anchors):
            # Extract the data for just that time interval -> Shape: (B, 1026)
            batch_t = batch[:, time_idx, :]
            
            batch_x_current = batch_t[:, :1024]
            t_current = (batch_t[:, 1024:1025] + 1) / 2 # Un-normalized due to mistake in data generation
            t_val = t_current[0].view(1, 1)
            
            labels = batch_t[:, 1025:]
                        
            with torch.no_grad():
                target_velocity = rf_dit(
                    batch_x_current.view(-1, 1, 32, 32), 
                    t_current.squeeze(-1), 
                    labels.long().squeeze(-1), 
                    # uncond_labels
                ).flatten(start_dim=1)
            
            batch_x_current.requires_grad_(True)

            # 2. Inject Time into PC Leaves
            # We condition the PC on the current time step
            mu, sigma = mlp(t_val)
            dynamic_params = torch.stack([mu.squeeze(), sigma.squeeze()], dim=-1).flatten()
            
            if isinstance(pc.input_layer_group[0].params, torch.nn.Parameter):
                del pc.input_layer_group[0].params
            pc.input_layer_group[0].params = dynamic_params

            # 3. Forward Pass (Log-Likelihood)
            with torch.cuda.device(device):
                lls = pc(batch_x_current, propagation_alg="LL")
                
                # logspace_flows=True ensures the node flows are saved in pc.node_mars
                pc.backward(batch_x_current, flows_memory=1.0, allow_modify_flows=False, propagation_alg="LL", logspace_flows=True)
                
            # 4. THE 5-LINE ANALYTICAL CHAIN RULE
            num_input_nodes = pc.input_layer_group[0].num_nodes
            num_vars = batch_x_current.size(1)
            K = num_input_nodes // num_vars # The number of Gaussian leaves per pixel
            batch_size = batch_x_current.size(0)

            # A. Extract per-sample structural flows (Detached, acting as fixed routing weights)
            # pc.node_mars shape is (Total_Nodes, Batch_Size)
            leaf_flows = pc.node_mars[:num_input_nodes, :].T.exp().detach() # Shape: (B, num_input_nodes)

            # B. Expand the 1024-D image to perfectly align with the leaf nodes
            x_expanded = batch_x_current.repeat_interleave(K, dim=1) # Shape: (B, num_input_nodes)

            # C. Calculate Gaussian derivatives (Autograd natively tracks mu and sigma here!)
            # Derivative of log N(x) = -(x - mu) / sigma^2
            grad_gaussian = -(x_expanded - mu.squeeze()) / (sigma.squeeze() ** 2 + 1e-6)

            # D. Chain Rule: Extract the exact SCORE
            exact_score = leaf_flows * grad_gaussian
            exact_score = exact_score.view(batch_size, num_vars, K).sum(dim=-1)

            # E. Corrected Algebraic Score-to-Velocity Conversion (Diffusion Convention)
            t_reshaped = t_current.view(-1, 1)
            
            # The singularity is at t=1.0, so we clamp the (1 - t) denominator
            denom_safe = torch.clamp(1.0 - t_reshaped, min=1e-3) 
            
            # v_t = -(x_t + t * Score) / (1 - t)
            predicted_velocity = -(batch_x_current + t_reshaped * exact_score) / denom_safe
            
            # F. Second Derivative: Joint Huber Backpropagation
            loss = torch.nn.functional.l1_loss(predicted_velocity, target_velocity)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pc.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Distributed Logging
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                avg_loss = loss.item() / world_size
                progress_bar.new_batch_done([avg_loss])
                if args.wandb:
                    wandb.log({"train/flow_mse": avg_loss, "trainer/step": step_count})
            
            step_count += 1

        # Epoch Checkpointing
        if rank == 0:
            avg_epoch_loss = progress_bar.epoch_ends()[0]
            with open(logfile, "a+") as f:
                f.write(f"[Epoch {epoch:05d}] - Flow MSE Loss: {avg_epoch_loss:.4f}\n")
            
            # Save the Neural-PC Hybrid
            juice.save(pcfile_last, pc)
            torch.save(mlp.module.state_dict(), pcfile_last.replace(".jpc", "_mlp.pt"))
            torch.save(optimizer.state_dict(), pcfile_last.replace(".jpc", "_optim.pt"))

    if rank == 0 and args.wandb:
        wandb.finish()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()
    args = parse_arguments()

    if args.port == 0:
        args.port = get_free_port()

    if world_size == 1:
        main(0, world_size, args)
    else:
        mp.spawn(main, args = (world_size, args), nprocs = world_size)