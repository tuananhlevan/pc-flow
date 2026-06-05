import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

from rec_flow.rf import RF
from rec_flow.dit import DiT_Llama

def generate_massive_trajectory_dataset(
    model, 
    cond_dataloader, 
    latent_shape, 
    anchor_points, 
    null_cond=None, 
    sample_steps=100, 
    cfg=2.0, 
    device="cuda"
):
    global_trajectory = {anchor: [] for anchor in anchor_points}
    
    # NEW: Create a list to store the labels aligned with the generated trajectories
    global_conditions = []
    
    print(f"Generating trajectories for {len(cond_dataloader.dataset)} samples...")
    
    for batch in tqdm(cond_dataloader, desc="ODE Integration"):
        cond_batch = batch[0]
        
        # Capture the label in CPU memory to avoid VRAM bloat
        global_conditions.append(cond_batch.clone().detach().cpu())
        
        cond_batch = cond_batch.to(device)
        b = cond_batch.size(0)
        
        z = torch.randn(b, *latent_shape, device=device)
        dt = 1.0 / sample_steps
        dt_tensor = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        
        batch_trajectory = {}
        
        if any(abs(1.0 - anchor) < 1e-5 for anchor in anchor_points):
            key = next(a for a in anchor_points if abs(1.0 - a) < 1e-5)
            batch_trajectory[key] = z.clone().detach().cpu()

        for i in range(sample_steps, 0, -1):
            t_current = i / sample_steps
            t = torch.tensor([t_current] * b).to(z.device)

            vc = model(z, t, cond_batch)
            
            if null_cond is not None:
                uncond_batch = torch.full_like(cond_batch, null_cond)
                vu = model(z, t, uncond_batch)
                vc = vu + cfg * (vc - vu)

            z = z - dt_tensor * vc
            t_next = (i - 1) / sample_steps
            
            for anchor in anchor_points:
                if t_next <= anchor < t_current and anchor not in batch_trajectory:
                    batch_trajectory[anchor] = z.clone().detach().cpu()
                    
        for anchor in anchor_points:
            global_trajectory[anchor].append(batch_trajectory[anchor])
            
    print("Assembling final dataset tensors...")
    for anchor in anchor_points:
        global_trajectory[anchor] = torch.cat(global_trajectory[anchor], dim=0)
        
    # NEW: Concatenate and return the labels alongside the trajectory
    global_conditions = torch.cat(global_conditions, dim=0)
        
    return global_trajectory, global_conditions


def create_joint_dataset(trajectory_dict, conditions, dir="pc_arena/data/TrajMNIST", splits=(0.8, 0.1, 0.1)):
    assert sum(splits) == 1.0, "Split ratios must sum to exactly 1.0"
    
    joint_samples = []
    
    for anchor_t, x_t_tensor in trajectory_dict.items():
        batch_size = x_t_tensor.size(0)
        x_t_flat = x_t_tensor.flatten(start_dim=1) 
        
        normalized_t = (anchor_t * 2.0) - 1.0
        
        t_col = torch.full((batch_size, 1), normalized_t, dtype=x_t_flat.dtype, device=x_t_flat.device)
        
        # NEW: Format the labels as a column to match the batch dimensions
        label_col = conditions.view(batch_size, 1).to(dtype=x_t_flat.dtype, device=x_t_flat.device)
        
        # NEW: Concatenate to make it 1026 dimensions (1024 pixels + 1 time + 1 label)
        x_joint = torch.cat([x_t_flat, t_col, label_col], dim=1)
        joint_samples.append(x_joint)
        
    full_joint_dataset = torch.stack(joint_samples, dim=1)
    
    indices = torch.randperm(full_joint_dataset.size(0))
    shuffled_dataset = full_joint_dataset[indices]
    
    total_samples = shuffled_dataset.size(0)
    train_size = int(splits[0] * total_samples)
    val_size = int(splits[1] * total_samples)
    
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]
    
    os.makedirs(dir, exist_ok=True)
    torch.save(train_dataset, os.path.join(dir, "train.pt"))
    torch.save(val_dataset, os.path.join(dir, "val.pt"))
    torch.save(test_dataset, os.path.join(dir, "test.pt"))
    
    print(f"Train (Joint): {train_dataset.shape} -> {os.path.join(dir, 'train.pt')}")


def create_paired_dataset(trajectory_dict, conditions, dir="pc_arena/data/PairedTrajMNIST", splits=(0.8, 0.1, 0.1)):
    sorted_anchors = sorted(list(trajectory_dict.keys()), reverse=True)
    paired_samples = []
    
    for i in range(len(sorted_anchors) - 1):
        t_curr = sorted_anchors[i]
        t_next = sorted_anchors[i+1]
        
        x_curr = trajectory_dict[t_curr].flatten(start_dim=1)
        x_next = trajectory_dict[t_next].flatten(start_dim=1)
        b = x_curr.size(0)
        t_col = torch.full((b, 1), float(t_curr), dtype=x_curr.dtype, device=x_curr.device)
        
        # NEW: Format the labels as a column
        label_col = conditions.view(b, 1).to(dtype=x_curr.dtype, device=x_curr.device)
        
        # NEW: Concatenate along dim=1 -> (B, 1024 + 1024 + 1 + 1) = (B, 2050)
        pair_tensor = torch.cat([x_curr, x_next, t_col, label_col], dim=1)
        paired_samples.append(pair_tensor)
        
    full_dataset = torch.stack(paired_samples, dim=1)
    
    indices = torch.randperm(full_dataset.size(0))
    shuffled_dataset = full_dataset[indices]
    
    total_samples = shuffled_dataset.size(0)
    train_size = int(splits[0] * total_samples)
    val_size = int(splits[1] * total_samples)
    
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]
    
    os.makedirs(dir, exist_ok=True)
    torch.save(train_dataset, os.path.join(dir, "train.pt"))
    torch.save(val_dataset, os.path.join(dir, "val.pt"))
    torch.save(test_dataset, os.path.join(dir, "test.pt"))
    
    print(f"Train (Paired): {train_dataset.shape} -> {os.path.join(dir, 'train.pt')}")


if __name__ == "__main__":
    channels = 1
    model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()
    model.load_state_dict(torch.load("contents/epoch_100/weight.pt"))
    
    model.eval()
    with torch.no_grad():
        total_samples = 60000
        labels = torch.randint(0, 10, (total_samples,))

        cond_dataset = TensorDataset(labels)
        cond_loader = DataLoader(cond_dataset, batch_size=512, shuffle=False)

        latent_shape = (1, 32, 32) 
        anchors = [1.0, 0.75, 0.5, 0.25, 0.0]
        
        # Unpack the returned conditions alongside the trajectory dict
        trajectory_dict, conditions = generate_massive_trajectory_dataset(
            model=model,
            cond_dataloader=cond_loader,
            latent_shape=latent_shape,
            anchor_points=anchors,
            null_cond=10,
            sample_steps=100, 
            cfg=2.0
        )
        
    # You can now call either generator and pass the conditions
    create_joint_dataset(trajectory_dict, conditions)