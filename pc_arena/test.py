import torch
import torch.nn as nn
import pyjuice as juice
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ---------------------------------------------------------
# 1. DEFINE THE TEMPORAL INJECTOR (Must match training exactly)
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
    
    base_dir = "logs/traj_mnist/[hclt_latent]-[flow-matching]/" 
    jpc_path = os.path.join(base_dir, "last.jpc")
    mlp_path = os.path.join(base_dir, "last_mlp.pt")

    # ---------------------------------------------------------
    # 2. LOAD THE SPATIAL PC BACKBONE
    # ---------------------------------------------------------
    print("Loading the Spatial PC Backbone...")
    root_ns = juice.load(jpc_path)
    pc = juice.compile(root_ns)
    pc.to(device)

    # Freeze structural parameters (inference only)
    for param in pc.parameters():
        param.requires_grad = False

    num_leaves = pc.input_layer_group[0].params.size(0) // 2 

    # ---------------------------------------------------------
    # 3. LOAD THE TEMPORAL MLP
    # ---------------------------------------------------------
    print("Loading the Temporal Flow MLP...")
    mlp = TimeMLP(num_leaves=num_leaves).to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp.eval()

    # ---------------------------------------------------------
    # 4. ODE FLOW MATCHING INFERENCE
    # ---------------------------------------------------------
    num_samples = 16
    latent_dim = 1024 # 32x32 image flattened
    sample_steps = 100
    dt = 1.0 / sample_steps

    # Start from pure unstructured Gaussian noise (t=1.0)
    print("Initializing unstructured noise...")
    x_t = torch.randn(num_samples, latent_dim, device=device)

    print("Solving Probability Flow ODE...")
    # Step backward from t=1.0 down to t=0.0
    for i in tqdm(range(sample_steps, 0, -1)):
        t_val = i / sample_steps
        
        # A. Predict Gaussian parameters for current time state
        with torch.no_grad():
            mu, sigma = mlp(torch.tensor([[t_val]], device=device))
        
        dynamic_params = torch.stack([mu.squeeze(), sigma.squeeze()], dim=-1).flatten().contiguous()
        
        # B. Inject parameters into PC
        # if isinstance(pc.input_layer_group[0].params, torch.nn.Parameter):
        #     del pc.input_layer_group[0].params
        # pc.input_layer_group[0].params = dynamic_params
        pc.input_layer_group[0].params.data.copy_(dynamic_params)

        # C. Forward/Backward Pass (Get exact structural routing in C++)
        with torch.cuda.device(device):
            lls = pc(x_t, propagation_alg="LL")
            pc.backward(x_t, flows_memory=1.0, allow_modify_flows=False, propagation_alg="LL", logspace_flows=True)
            
        # ---------------------------------------------------------
        # D. THE 5-LINE ANALYTICAL CHAIN RULE (Score Extraction)
        # ---------------------------------------------------------
        num_input_nodes = pc.input_layer_group[0].num_nodes
        K = num_input_nodes // latent_dim 
        
        # 1. Extract structural flows natively from C++ memory
        leaf_flows = pc.node_mars[:num_input_nodes, :].T.exp().detach() # Shape: (B, num_input_nodes)
        
        # 2. Expand pixels to align with Gaussian leaves
        x_expanded = x_t.repeat_interleave(K, dim=1)
        
        # 3. Calculate exact Gaussian derivatives: -(x - mu) / sigma^2
        grad_gaussian = -(x_expanded - mu.squeeze()) / (sigma.squeeze() ** 2 + 1e-6)
        
        # 4. Chain Rule Integration
        exact_score_expanded = leaf_flows * grad_gaussian
        
        # 5. Sum across K leaves to get the final exact 1024-D Score
        score = exact_score_expanded.view(num_samples, latent_dim, K).sum(dim=-1)
        # ---------------------------------------------------------
        
        # E. Convert Score to Target Velocity (Tweedie Math)
        # Clamp t to prevent catastrophic division by zero at the final step
        denom_safe = max(1.0 - t_val, 1e-3) 
        
        # v_t = -(x_t + t * Score) / (1 - t)
        velocity = -(x_t + t_val * score) / denom_safe
        
        # F. Euler Step (remains exactly the same)
        x_t = x_t - velocity * dt

    print("Flow trajectory complete!")

    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    final_images = x_t.view(-1, 1, 32, 32).cpu().detach()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        # Values clamped strictly for visual fidelity of the digits
        ax.imshow(final_images[i][0], cmap='gray', vmin=-1, vmax=1) 
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()