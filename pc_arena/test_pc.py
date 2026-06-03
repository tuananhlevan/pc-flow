import torch
import torch.nn as nn
import pyjuice as juice
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. DEFINE THE TEMPORAL INJECTOR 
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
        
    def forward(self, t):
        out = self.net(t)
        mu, log_var = out.chunk(2, dim=-1)
        sigma = torch.exp(0.5 * log_var) + 1e-4 
        return mu, sigma

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ⚠️ REPLACE THESE PATHS with your actual log directory
    base_dir = "logs/traj_mnist/[hclt_latent]-[flow-matching]/" 
    jpc_path = os.path.join(base_dir, "last.jpc")
    mlp_path = os.path.join(base_dir, "last_mlp.pt")

    # ---------------------------------------------------------
    # 2. LOAD MODELS
    # ---------------------------------------------------------
    print("Loading the Spatial PC Backbone...")
    root_ns = juice.load(jpc_path)
    pc = juice.compile(root_ns)
    pc.to(device)

    num_leaves = pc.input_layer_group[0].params.size(0) // 2 

    print("Loading the Temporal Flow MLP...")
    mlp = TimeMLP(num_leaves=num_leaves).to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp.eval()

    # ---------------------------------------------------------
    # 3. ONE-STEP UNCONDITIONAL DECODING
    # ---------------------------------------------------------
    num_samples = 16
    latent_dim = 1024 # 32x32 flattened
    
    # A. Query the MLP for the clean data manifold (t = 0.0)
    print("Decoding the t=0.0 parameter state...")
    with torch.no_grad():
        t_target = torch.zeros(1, 1, device=device) # Target t=0.0
        mu, sigma = mlp(t_target)
        
    dynamic_params = torch.stack([mu.squeeze(), sigma.squeeze()], dim=-1).flatten().contiguous()
    
    # B. Lock the decoded parameters into the PC safely
    # We use in-place .copy_() to ensure the PyJuice C++ pointers remain perfectly intact
    pc.input_layer_group[0].params.data.copy_(dynamic_params)

    # C. Instant Unconditional Ancestral Sampling
    print("Executing one-step ancestral sampling...")
    with torch.cuda.device(device):
        # We bypass the forward pass and missing mask completely.
        # Since the PC is now mathematically locked to t=0.0, we just sample unconditionally.
        generated_latents = juice.queries.sample(pc, num_samples=num_samples, conditional=False)

    print("Generation complete!")

    # ---------------------------------------------------------
    # 4. VISUALIZATION
    # ---------------------------------------------------------
    final_images = generated_latents.view(-1, 1, 32, 32).cpu().detach()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(final_images[i][0], cmap='gray', vmin=-1, vmax=1) 
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()