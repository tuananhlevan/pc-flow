import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

from rec_flow.rf import RF
from rec_flow.dit import DiT_Llama

def generate_teacher_dataset(
    model, 
    cond_dataloader, 
    latent_shape, 
    null_cond=None, 
    sample_steps=100, 
    cfg=2.0, 
    device="cuda"
):
    """
    Uses the Teacher DiT to generate the final clean images (t=0.0).
    """
    clean_images = []
    
    print(f"Generating clean teacher targets for {len(cond_dataloader.dataset)} samples...")
    
    for batch in tqdm(cond_dataloader, desc="ODE Integration"):
        cond_batch = batch[0].to(device)
        b = cond_batch.size(0)
        
        # In your DiT formulation, t=1.0 is pure noise
        z = torch.randn(b, *latent_shape, device=device)
        
        dt = 1.0 / sample_steps
        dt_tensor = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        
        for i in range(sample_steps, 0, -1):
            t_current = i / sample_steps
            t = torch.tensor([t_current] * b).to(z.device)

            # Velocity prediction with Classifier-Free Guidance
            vc = model(z, t, cond_batch)
            
            if null_cond is not None:
                uncond_batch = torch.full_like(cond_batch, null_cond)
                vu = model(z, t, uncond_batch)
                vc = vu + cfg * (vc - vu)

            # Euler step
            z = z - dt_tensor * vc
            
        # At the end of the loop, z is the clean image at t=0.0
        # Flatten the (B, 1, 32, 32) images to (B, 1024)
        z_flat = z.clone().detach().cpu().flatten(start_dim=1)
        clean_images.append(z_flat)
            
    print("Assembling final dataset tensor...")
    full_dataset = torch.cat(clean_images, dim=0)
    
    # Shuffle to ensure heterogeneous mini-batches
    indices = torch.randperm(full_dataset.size(0))
    return full_dataset[indices]

def split_and_save(dataset, dir="pc_arena/data/DistilledMNIST", splits=(0.8, 0.1, 0.1)):
    """
    Splits the pure (N, 1024) dataset into Train, Val, and Test sets.
    """
    assert sum(splits) == 1.0, "Split ratios must sum to exactly 1.0"
    
    total_samples = dataset.size(0)
    train_size = int(splits[0] * total_samples)
    val_size = int(splits[1] * total_samples)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]
    
    os.makedirs(dir, exist_ok=True)
    
    train_path = os.path.join(dir, "train.pt")
    val_path = os.path.join(dir, "val.pt")
    test_path = os.path.join(dir, "test.pt")
    
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)
    
    print(f"Total distilled samples: {total_samples}")
    print(f"Train: {train_dataset.shape} -> {train_path}")
    print(f"Val:   {val_dataset.shape} -> {val_path}")
    print(f"Test:  {test_dataset.shape} -> {test_path}")
    
if __name__ == "__main__":
    channels = 1
    model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()
    
    # Load your pre-trained Teacher weights
    model.load_state_dict(torch.load("contents/epoch_100/weight.pt"))
    model.eval()
    
    with torch.no_grad():
        total_samples = 30000
        labels = torch.randint(0, 10, (total_samples,))

        cond_dataset = TensorDataset(labels)
        cond_loader = DataLoader(cond_dataset, batch_size=512, shuffle=False)

        latent_shape = (1, 32, 32) 
        
        full_dataset = generate_teacher_dataset(
            model=model,
            cond_dataloader=cond_loader,
            latent_shape=latent_shape,
            null_cond=10,
            sample_steps=100,  
            cfg=2.0
        )
        
    split_and_save(full_dataset, dir="pc_arena/data/DistilledMNIST", splits=(0.8, 0.1, 0.1))