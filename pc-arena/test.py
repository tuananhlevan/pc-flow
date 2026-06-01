import pyjuice as juice
import torch
import matplotlib.pyplot as plt

# 1. Load and compile
root_ns = juice.load("logs/traj_mnist/[hclt_latent]-[anemone]-[pseudocount-1e-6]/last.jpc")
pc = juice.compile(root_ns)
pc.to(torch.device("cuda:0"))

num_samples = 16
total_vars = 2048

# 2. Setup Data and Missing Mask
data = torch.zeros((num_samples, total_vars), dtype=torch.float32, device="cuda")
missing_mask = torch.ones(total_vars, dtype=torch.bool, device="cuda")

# 3. Clamp the entire Time Channel
# The first 1024 variables are pixels (missing). 
# The last 1024 variables are the duplicated time states (evidence).
data[:, 1024:] = 1.0 
missing_mask[1024:] = False 

# 4. Propagate the massive evidence signal
print("Flooding the PC with global time evidence...")
lls = pc(data, missing_mask=missing_mask)

# 5. Sample the pixels
print("Generating sharp latents...")
generated_latents = juice.queries.sample(pc, conditional=True)

# 6. Extract just the pixel channel and reshape to view
final_images = generated_latents[:, :1024].view(-1, 1, 32, 32).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(final_images[i][0], cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')
plt.show()