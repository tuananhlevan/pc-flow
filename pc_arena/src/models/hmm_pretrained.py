import torch
import pyjuice as juice

def hmm_pretrained(seq_length, ckpt_fname):

    checkpoint = torch.load(ckpt_fname, map_location="cpu", weights_only=True)
    hmm_state_dict = checkpoint['decoder_state_dict']
    alpha_logits = hmm_state_dict['alpha_logits']
    beta_logits = hmm_state_dict['beta_logits']
    gamma_logits = hmm_state_dict['gamma_logits']
    
    alpha = torch.softmax(alpha_logits, dim=-1)
    beta = torch.softmax(beta_logits, dim=-1)
    gamma = torch.softmax(gamma_logits, dim=-1).squeeze(0)

    num_latents = alpha.size(0)
    num_emits = beta.size(1)

    root_ns = juice.structures.HMM(
        seq_length = seq_length,
        num_latents = num_latents,
        num_emits = num_emits,
        alpha = alpha,
        beta = beta,
        gamma = gamma
    )

    return root_ns

def hmm_pretrained_lvd(seq_length, ckpt_fname):
    checkpoint = torch.load(ckpt_fname, map_location="cpu")

    alpha = torch.exp(checkpoint['alpha'])
    beta = torch.exp(checkpoint['beta'])
    gamma = torch.exp(checkpoint['gamma'])

    num_latents = checkpoint['hidden_states']
    num_emits = checkpoint['vocab_size']

    root_ns = juice.structures.HMM(
        seq_length=seq_length,
        num_latents=num_latents,
        num_emits=num_emits,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    return root_ns