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
    """
    Wraps the trajectory generation across an entire dataset/dataloader.
    
    Args:
        model: Your Flow Matching Teacher model.
        cond_dataloader: PyTorch DataLoader yielding conditioning tensors (e.g., labels).
        latent_shape: The shape of a single noise tensor (e.g., (1, 32, 32) or (1024,)).
        anchor_points: List of discrete timesteps to save.
        null_cond: Tensor for Classifier-Free Guidance.
        sample_steps: Number of integration steps.
        cfg: Guidance scale.
        device: Target compute device.
    """
    # 1. Initialize a global dictionary to hold lists of tensors
    # We use lists because appending to lists is O(1). 
    # Using torch.cat() inside a loop causes O(N^2) memory reallocation and will crash your RAM.
    global_trajectory = {anchor: [] for anchor in anchor_points}
    
    print(f"Generating trajectories for {len(cond_dataloader.dataset)} samples...")
    
    # 2. The Outer Loop (Iterating over batches)
    for batch in tqdm(cond_dataloader, desc="ODE Integration"):
        
        # Move condition to device and determine current batch size
        cond_batch = batch[0]
        
        # Now you can safely move it to the device
        cond_batch = cond_batch.to(device)
        b = cond_batch.size(0)
        
        # Sample fresh base noise for this specific batch (t=1.0)
        z = torch.randn(b, *latent_shape, device=device)
        
        # --- START OF YOUR ODE LOGIC ---
        dt = 1.0 / sample_steps
        dt_tensor = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        
        batch_trajectory = {}
        
        # Capture t=1.0
        if any(abs(1.0 - anchor) < 1e-5 for anchor in anchor_points):
            key = next(a for a in anchor_points if abs(1.0 - a) < 1e-5)
            batch_trajectory[key] = z.clone().detach().cpu()

        for i in range(sample_steps, 0, -1):
            t_current = i / sample_steps
            t = torch.tensor([t_current] * b).to(z.device)

            # Velocity prediction with Classifier-Free Guidance
            vc = model(z, t, cond_batch)
            
            if null_cond is not None:
                # Dynamically create an uncond batch of the same shape/device as cond_batch
                uncond_batch = torch.full_like(cond_batch, null_cond)
                
                vu = model(z, t, uncond_batch)
                vc = vu + cfg * (vc - vu)

            # Euler step
            z = z - dt_tensor * vc
            
            # Temporal location
            t_next = (i - 1) / sample_steps
            
            # Intercept anchors
            for anchor in anchor_points:
                # Using the threshold logic we established to guarantee capture
                if t_next <= anchor < t_current and anchor not in batch_trajectory:
                    batch_trajectory[anchor] = z.clone().detach().cpu()
        # --- END OF YOUR ODE LOGIC ---
                    
        # 3. Append this batch's isolated results to the global dictionary
        for anchor in anchor_points:
            global_trajectory[anchor].append(batch_trajectory[anchor])
            
    # 4. Final Memory Assembly
    # Concatenate the lists of tensors into massive single tensors all at once
    print("Assembling final dataset tensors...")
    for anchor in anchor_points:
        global_trajectory[anchor] = torch.cat(global_trajectory[anchor], dim=0)
        
    return global_trajectory

def create_joint_dataset(trajectory_dict, dir="pc-arena/data/TrajMNIST", splits=(0.8, 0.1, 0.1)):
    """
    Creates a joint space-time dataset and splits it into Train, Val, and Test sets.
    
    Args:
        trajectory_dict: Dictionary of generated trajectories {t_anchor: tensor}.
        dir: Directory to save the output files.
        splits: Tuple of (train_ratio, val_ratio, test_ratio). Must sum to 1.0.
    """
    assert sum(splits) == 1.0, "Split ratios must sum to exactly 1.0"
    
    joint_samples = []
    
    for anchor_t, x_t_tensor in trajectory_dict.items():
        batch_size = x_t_tensor.size(0)
        
        # Flatten the 4D image tensor (B, C, H, W) down to 2D (B, 1024)
        x_t_flat = x_t_tensor.flatten(start_dim=1) 
        
        # Normalize t from [0, 1] to [-1, 1]
        normalized_t = (anchor_t * 2.0) - 1.0
        
        # Create the t column: shape (batch_size, 1)
        t_col = torch.full((batch_size, 1), normalized_t, dtype=x_t_flat.dtype, device=x_t_flat.device)
        
        # Concatenate to make it 1025 dimensions (batch_size, 1025)
        x_joint = torch.cat([x_t_flat, t_col], dim=1)
        joint_samples.append(x_joint)
        
    # Stack all anchors together
    full_joint_dataset = torch.cat(joint_samples, dim=0)
    
    # SHUFFLE to prevent catastrophic forgetting
    indices = torch.randperm(full_joint_dataset.size(0))
    shuffled_dataset = full_joint_dataset[indices]
    
    # --- SPLITTING LOGIC ---
    total_samples = shuffled_dataset.size(0)
    train_size = int(splits[0] * total_samples)
    val_size = int(splits[1] * total_samples)
    
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]
    
    # --- SAVING LOGIC ---
    os.makedirs(dir, exist_ok=True)
    
    train_path = os.path.join(dir, "train.pt")
    val_path = os.path.join(dir, "val.pt")
    test_path = os.path.join(dir, "test.pt")
    
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)
    
    print(f"Total joint samples: {total_samples}")
    print(f"Train: {train_dataset.shape} -> {train_path}")
    print(f"Val:   {val_dataset.shape} -> {val_path}")
    print(f"Test:  {test_dataset.shape} -> {test_path}")
    
def create_paired_dataset(trajectory_dict, dir="pc-arena/data/PairedTrajMNIST", splits=(0.8, 0.1, 0.1)):
    # Ensure anchors are sorted descending: [1.0, 0.75, 0.5, 0.25, 0.0]
    sorted_anchors = sorted(list(trajectory_dict.keys()), reverse=True)
    
    paired_samples = []
    
    # Iterate through adjacent pairs (e.g., 1.0 -> 0.75)
    for i in range(len(sorted_anchors) - 1):
        t_curr = sorted_anchors[i]
        t_next = sorted_anchors[i+1]
        
        x_curr = trajectory_dict[t_curr].flatten(start_dim=1)
        x_next = trajectory_dict[t_next].flatten(start_dim=1)
        
        b = x_curr.size(0)
        t_col = torch.full((b, 1), float(t_curr), dtype=x_curr.dtype, device=x_curr.device)
        
        # Concatenate along dim=1 -> (B, 1024 + 1024 + 1) = (B, 2049)
        pair_tensor = torch.cat([x_curr, x_next, t_col], dim=1)
        paired_samples.append(pair_tensor)
        
    full_dataset = torch.cat(paired_samples, dim=0)
    
    indices = torch.randperm(full_dataset.size(0))
    shuffled_dataset = full_dataset[indices]
    
    # --- SPLITTING LOGIC ---
    total_samples = shuffled_dataset.size(0)
    train_size = int(splits[0] * total_samples)
    val_size = int(splits[1] * total_samples)
    
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size : train_size + val_size]
    test_dataset = shuffled_dataset[train_size + val_size :]
    
    # Shuffle to ensure mixed batches
    indices = torch.randperm(full_dataset.size(0))
    shuffled_dataset = full_dataset[indices]
    
    os.makedirs(dir, exist_ok=True)
    train_path = os.path.join(dir, "train.pt")
    val_path = os.path.join(dir, "val.pt")
    test_path = os.path.join(dir, "test.pt")
    
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)
    
    print(f"Total joint samples: {total_samples}")
    print(f"Train: {train_dataset.shape} -> {train_path}")
    print(f"Val:   {val_dataset.shape} -> {val_path}")
    print(f"Test:  {test_dataset.shape} -> {test_path}")
    
    print(f"Saved Paired Trajectories: {shuffled_dataset.shape} -> {train_path}")    

if __name__ == "__main__":
    channels = 1
    model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()
    model.load_state_dict(torch.load("contents/epoch_100/weight.pt"))
    
    model.eval()
    with torch.no_grad():

        # 1. Generate 60,000 random labels between 0 and 9
        # (Or you can load the actual MNIST training labels if you prefer)
        total_samples = 60000
        labels = torch.randint(0, 10, (total_samples,))

        # 2. Wrap it in a Dataset and DataLoader
        cond_dataset = TensorDataset(labels)
        cond_loader = DataLoader(cond_dataset, batch_size=512, shuffle=False)

        latent_shape = (1, 32, 32) 
        anchors = [1.0, 0.75, 0.5, 0.25, 0.0]
        
        trajectory_dict = generate_massive_trajectory_dataset(
            model=model,
            cond_dataloader=cond_loader,
            latent_shape=latent_shape,
            anchor_points=anchors,
            null_cond=10,
            sample_steps=100,  # 100 steps aligns perfectly with quarters (0.75, 0.5, etc)
            cfg=2.0
        )
        
    create_paired_dataset(trajectory_dict, dir="pc_arena/data/TrajMNIST")