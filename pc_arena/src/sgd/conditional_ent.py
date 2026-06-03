import torch
import triton
import triton.language as tl
import pyjuice as juice
import time
import math

# In the latest triton, math functions were shuffled around into different modules:
# https://github.com/openai/triton/pull/3172
if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math

from pyjuice.model.backend import sgd_par_update, normalize_parameters, compute_cum_par_flows


##################
## Forward pass ##
##################


@torch.compile
def update_fn(nmars, cmars, params, cents, node_ents, sid, eid):
    cond_params = cmars[:,None,:] + params[:,:,None].log() - nmars[None,:,:]

    nents = (cond_params.exp() * (cents[:,None,:] - cond_params.clamp(min = -1000.0))).sum(dim = 0)

    node_ents[sid:eid,:] += nents


def sum_layer_cond_ent(layer, pc, node_ents, element_ents):
    block_size = layer.block_size
    for nids, cids, pids, pfids, ch_block_size in zip(layer.partitioned_nids, layer.partitioned_cids, layer.partitioned_pids, layer.partitioned_pfids, layer.cs_block_sizes):
        n_blk_params = block_size * ch_block_size
        for i in range(nids.size(0)):
            nmars = pc.node_mars[nids[i]:nids[i]+block_size,:]
            node_ents[nids[i]:nids[i]+block_size,:] = 0.0
            for j in range(0, cids.size(1), ch_block_size):
                params = pc.params[pids[i,j]:pids[i,j]+n_blk_params].reshape(ch_block_size, block_size)
                cmars = pc.element_mars[cids[i,j]:cids[i,j]+ch_block_size,:]

                cents = element_ents[cids[i,j]:cids[i,j]+ch_block_size,:]

                update_fn(nmars, cmars, params, cents, node_ents, nids[i], nids[i]+block_size)


@triton.jit
def sum_layer_cond_ent_block_kernel(node_ents, element_ents, node_mars, element_mars, params, nids, cids_start, cids_increment, 
                                    pids_start, pids_increment, batch_size, BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                    K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Node offsets
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

    # Edge offsets
    offs_edge = tl.arange(0, TILE_SIZE_K)

    # Initialize pointers to `params`
    offs_estart = nblock_id * TILE_SIZE_K + offs_edge
    offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
    par_start = tl.load(pids_start + offs_estart)
    epars_ptr = params + \
        offs_node[:,None] + \
        par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars` and `element_ents`
    edge_start = tl.load(cids_start + offs_estart)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    eents_ptr = element_ents + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Node mars
    off_nids = tl.load(nids + nblock_id)
    offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nmars = tl.load(node_mars + offs_nmars, mask = mask_batch[None,:])

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        eents = tl.load(eents_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        # epars * exp(emars) * eents / exp(nmars)
        val1 = emars + tl.log(eents) # [TILE_SIZE_K, BLOCK_B]
        emars_max = tl.max(emars, axis = 0)[None,:]
        val1_sub = tl.where(emars_max != -float("inf"), tl.exp(val1 - emars_max), 0.0)

        epars_bf16 = epars.to(tl.float32)
        val1_bf16 = val1_sub.to(tl.float32)
        out1 = tl.dot(epars_bf16, val1_bf16).to(tl.float32)
        acc += out1 * tl.exp(emars_max - nmars)

        # -epars * log(epars) * exp(emars) / exp(nmars)
        val2 = epars * tl.log(epars)
        emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

        val2_bf16 = val2.to(tl.float32)
        emars_bf16 = emars_sub.to(tl.float32)
        out2 = tl.dot(val2_bf16, emars_bf16).to(tl.float32)
        acc -= out2 * tl.exp(emars_max - nmars)

        # -epars * emars * exp(emars) / exp(nmars)
        val3_sub = emars_sub * emars
        out3 = tl.dot(epars, val3_sub)
        acc -= out3 * tl.exp(emars_max - nmars)

        # epars * emars * exp(nmars) / exp(nmars)
        out4 = tl.dot(epars_bf16, emars_bf16).to(tl.float32)
        acc += out4 * tl.exp(emars_max - nmars) * nmars

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment `emars_ptr` and `eents_ptr`
        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[:,None] * batch_size
        eents_ptr += cids_inc[:,None] * batch_size
        cids_inc_ptr += TILE_SIZE_K

    acc = tl.where(acc < 0.0, 1e-12, acc)

    tl.store(node_ents + offs_nmars, acc, mask = mask_batch[None,:])


def sum_layer_cond_ent_triton(layer, pc, node_ents, element_ents):

    params = pc.params
    node_mars = pc.node_mars
    element_mars = pc.element_mars
    
    for partition_id in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[partition_id]
        cids = layer.partitioned_cids[partition_id]
        pids = layer.partitioned_pids[partition_id]
        pfids = layer.partitioned_pfids[partition_id]

        block_size = layer.block_size
        num_nblocks = nids.size(0)
        layer_n_nodes = num_nblocks * block_size
        num_edges = cids.size(1)
        batch_size = node_ents.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        base_size = min(block_size, num_edges, BATCH_SIZE_NP2, 128)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(2048 // remainder, base_size * remainder, num_edges)
        TILE_SIZE_M = min(2048 // TILE_SIZE_K, block_size)
        BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        signature = ("block_sparse", partition_id, TILE_SIZE_K)
        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16 and signature in layer._cached_fw_pcids:
            cids_start, cids_increment, pids_start, pids_increment = layer._cached_fw_pcids[signature]

            BLOCK_SIZE_M = block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            sum_layer_cond_ent_block_kernel[grid](
                node_ents, 
                element_ents, 
                node_mars,
                element_mars,
                params, 
                nids, 
                cids_start,
                cids_increment, 
                pids_start,
                pids_increment,
                batch_size,
                BLOCK_B = BLOCK_B,
                TILE_SIZE_K = TILE_SIZE_K,
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M,
                BLOCK_SIZE_M = BLOCK_SIZE_M
            )

        elif block_size == 1 and nids.size(0) == 1:
            nmars = pc.node_mars[nids[0],:]
            
            params = pc.params[pids[0,:]]
            cmars = pc.element_mars[cids[0,:],:]
            cents = element_ents[cids[0,:],:]

            cond_params = cmars + params[:,None].log() - nmars[None,:]
            nents = (cond_params.exp() * (cents - cond_params.clamp(min = -1000.0))).sum(dim = 0)

            node_ents[nids[0],:] = nents

        else:
            raise NotImplementedError()


def compute_conditional_ent(pc, x, ret_node_ents = False, forward_pass_completed = False):

    with torch.no_grad():

        # First do the normal forward pass (only if the forward pass is not executed externally)
        if not forward_pass_completed:
            lls = pc(x, propagation_alg = "LL", force_use_fp32 = False)

        # Initialize buffers for conditional ent
        node_ents = torch.zeros_like(pc.node_mars)
        element_ents = torch.zeros_like(pc.element_mars)

        # The CondENT forward pass
        # We do nothing for the input layers as they have 0 CondENT if full evidence is provided
        for layer_id, layer_group in enumerate(pc.inner_layer_groups):
            if layer_group.is_prod():
                # Prod layer
                layer_group(node_ents, element_ents)
                layer_group(pc.node_mars, pc.element_mars)

            elif layer_group.is_sum():
                # Sum layer
                for layer in layer_group:
                    # node_ents_gt = node_ents.clone()
                    # sum_layer_cond_ent(layer, pc, node_ents_gt, element_ents)

                    sum_layer_cond_ent_triton(layer, pc, node_ents, element_ents)

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")

    assert pc._root_node_range[1] - pc._root_node_range[0] == 1
    if ret_node_ents:
        return lls[:,0], node_ents[pc._root_node_range[0],:], node_ents
    else:
        return lls[:,0], node_ents[pc._root_node_range[0],:]


###################
## Backward pass ##
###################


@torch.compile
def update_fn2(cmars, params, nmars, nentgrads, cents, node_flows, element_flows, param_flows, nids, cids, pfids, 
               n_blk_params, block_size, ch_block_size, i, j, lamda):
    cond_params = cmars[:,None,:] + params[:,:,None].log() - nmars[None,:,:]

    # # Gradient w.r.t. log(theta)
    # pgrads = (nentgrads[None,:,:] * cond_params.exp() * (cents[:,None,:] - cond_params - 1)).sum(dim = 2).reshape(-1)
    # param_flows[pfids[i,j]:pfids[i,j]+n_blk_params] -= pgrads * lamda # Subtract because we want to minimize conditional entropy

    # # Gradient w.r.t. log(nmars)
    # ngrads = (nentgrads[None,:,:] * cond_params.exp() * (cents[:,None,:] - cond_params - 1)).sum(dim = 0)
    # node_flows[nids[i]:nids[i]+block_size] += ngrads * lamda # The two negations from "minimizing" conditional entropy and from the negation in `cond_params` cancel out 

    # Gradient w.r.t. log(cmars)
    cgrads = (nentgrads[None,:,:] * cond_params.exp() * (cents[:,None,:] - cond_params - 1)).sum(dim = 1)
    element_flows[cids[i,j]:cids[i,j]+ch_block_size] -= cgrads * lamda # Subtract because we want to minimize conditional entropy


def sum_layer_cond_ent_grad_params(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda):

    block_size = layer.block_size
    for nids, cids, pids, pfids, ch_block_size in zip(layer.partitioned_nids, layer.partitioned_cids, layer.partitioned_pids, layer.partitioned_pfids, layer.cs_block_sizes):
        n_blk_params = block_size * ch_block_size
        for i in range(nids.size(0)):
            nmars = pc.node_mars[nids[i]:nids[i]+block_size,:]
            nentgrads = node_ent_grads[nids[i]:nids[i]+block_size,:]
            for j in range(0, cids.size(1), ch_block_size):
                params = pc.params[pids[i,j]:pids[i,j]+n_blk_params].reshape(ch_block_size, block_size)
                cmars = pc.element_mars[cids[i,j]:cids[i,j]+ch_block_size,:]
                cents = element_ents[cids[i,j]:cids[i,j]+ch_block_size,:]

                update_fn2(cmars, params, nmars, nentgrads, cents, pc.node_flows, pc.element_flows, pc.param_flows, 
                           nids, cids, pfids, n_blk_params, block_size, ch_block_size, i, j, lamda)


@triton.jit
def sum_layer_cond_ent_grad_params_kernel1(node_flows, node_mars, element_mars, element_ents, node_ent_grads, params, 
                                           param_flows, nids, cids, pids, pfids, lamda, batch_size, num_edges: tl.constexpr, 
                                           TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                           TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Batch offsets and mask
    offs_batch = tl.arange(0, TILE_SIZE_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars` and `element_ents`
    offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
    edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
    emars_ptr = element_mars + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]
    eents_ptr = element_ents + \
        edge_start[None,:] * batch_size + \
        offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

    # Initialize pointers to `node_mars` and `node_ent_grads`
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    off_nids = tl.load(nids + nblock_id)
    nmars_ptr = node_mars + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nentgrads_ptr = node_ent_grads + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

    # Get `epars`
    par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
    epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
    epars = tl.load(params + epars_offsets)

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)

    for b in range(0, B_NUM_TILES):
        emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]
        eents = tl.load(eents_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]
        nentgrads = tl.load(nentgrads_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]

        ## pgrads = (nentgrads[None,:,:] * cond_params.exp() * (cents[:,None,:] - cond_params - 1)).sum(dim = 2).reshape(-1)

        # nentgrads * epars * exp(emars) * eents / exp(nmars)
        val1 = emars + tl.log(eents) # [TILE_SIZE_B, TILE_SIZE_K]
        val1_max = tl.max(val1, axis = 1)
        val1_sub = tl.where(val1_max[:,None] != -float("inf"), tl.exp(val1 - val1_max[:,None]), 0.0)
        val2_sub = nentgrads * tl.exp(val1_max[None,:] - nmars) # [TILE_SIZE_M, TILE_SIZE_B]

        val1_bf16 = val1_sub.to(tl.float32)
        val2_bf16 = val2_sub.to(tl.float32)
        out1 = tl.dot(val2_bf16, val1_bf16).to(tl.float32)
        acc += out1 * epars

        # -nentgrads * epars * (log(epars) + 1) * exp(emars) / exp(nmars)
        emars_max = tl.max(emars, axis = 1)
        emars_sub = tl.where(emars_max[:,None] != -float("inf"), tl.exp(emars - emars_max[:,None]), 0.0) # [TILE_SIZE_B, TILE_SIZE_K]
        nmars_sub = nentgrads * tl.exp(emars_max[None,:] - nmars) # [TILE_SIZE_M, TILE_SIZE_B]

        nmars_bf16 = nmars_sub.to(tl.float32)
        emars_bf16 = emars_sub.to(tl.float32)
        out2 = tl.dot(nmars_bf16, emars_bf16).to(tl.float32)
        acc -= out2 * epars * (tl.log(epars) + 1.0)

        # -nentgrads * epars * emars * exp(emars) / exp(nmars)
        val3_sub = emars_sub * emars # [TILE_SIZE_B, TILE_SIZE_K]
        out3 = tl.dot(nmars_sub, val3_sub)
        acc -= out3 * epars

        # nentgrads * epars * emars * nmars / exp(nmars)
        val4_sub = nentgrads * tl.exp(emars_max[None,:] - nmars) * nmars # [TILE_SIZE_M, TILE_SIZE_B]

        val4_bf16 = val4_sub.to(tl.float32)
        out4 = tl.dot(val4_bf16, emars_bf16).to(tl.float32)
        acc += out4 * epars

        # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
        emars_ptr += TILE_SIZE_B
        eents_ptr += TILE_SIZE_B
        nmars_ptr += TILE_SIZE_B
        nentgrads_ptr += TILE_SIZE_B

        # Update batch mask
        offs_batch += TILE_SIZE_B
        mask_batch = offs_batch < batch_size

    acc = -acc * lamda

    parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
    eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    tl.atomic_add(param_flows + eparflows_offsets, acc)


def sum_layer_cond_ent_grad_params_triton1(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda):

    params = pc.params
    param_flows = pc.param_flows
    node_mars = pc.node_mars
    element_mars = pc.element_mars
    node_flows = pc.node_flows
    element_flows = pc.element_flows
    
    for partition_id in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[partition_id]
        cids = layer.partitioned_cids[partition_id]
        pids = layer.partitioned_pids[partition_id]
        pfids = layer.partitioned_pfids[partition_id]

        num_nblocks = nids.size(0)
        block_size = layer.block_size
        layer_n_nodes = num_nblocks * block_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(block_size, num_edges, BATCH_SIZE_NP2)
        if base_size >= 64:
            TILE_SIZE_B = min(2048 // 32, BATCH_SIZE_NP2)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_B = min(2048 // remainder, base_size * remainder, BATCH_SIZE_NP2)
        TILE_SIZE_M = min(2048 // TILE_SIZE_B, block_size)
        TILE_SIZE_K = min(2048 // TILE_SIZE_B, num_edges)

        B_NUM_TILES = batch_size // TILE_SIZE_B

        if TILE_SIZE_M >= 16 and TILE_SIZE_B >= 16 and TILE_SIZE_K >= 16:

            grid = (triton.cdiv(num_edges, TILE_SIZE_K), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            sum_layer_cond_ent_grad_params_kernel1[grid](
                node_flows = node_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                element_ents = element_ents,
                node_ent_grads = node_ent_grads,
                params = params, 
                param_flows = param_flows, 
                nids = nids, 
                cids = cids, 
                pids = pids,
                pfids = pfids,
                lamda = lamda,
                batch_size = batch_size, 
                num_edges = num_edges, 
                TILE_SIZE_B = TILE_SIZE_B, 
                B_NUM_TILES = B_NUM_TILES, 
                TILE_SIZE_K = TILE_SIZE_K, 
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = block_size
            )
        elif block_size == 1 and nids.size(0) == 1:
            nmars = pc.node_mars[nids[0],:]
            nentgrads = node_ent_grads[nids[0],:]
            
            params = pc.params[pids[0,:]]
            cmars = pc.element_mars[cids[0,:],:]
            cents = element_ents[cids[0,:],:]

            cond_params = cmars + params[:,None].log() - nmars[None,:]
            pgrads = (nentgrads[None,:] * cond_params.exp() * (cents - cond_params - 1)).sum(dim = 1)

            pc.param_flows[pfids[0,:]] -= pgrads * lamda


        else:
            raise NotImplementedError()


@triton.jit
def sum_layer_cond_ent_grad_params_kernel2(node_ents, element_ents, node_mars, element_mars, node_flows, node_ent_grads, params, nids, cids_start, cids_increment, 
                                           pids_start, pids_increment, batch_size, lamda, BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                           K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Node offsets
    offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

    # Edge offsets
    offs_edge = tl.arange(0, TILE_SIZE_K)

    # Initialize pointers to `params`
    offs_estart = nblock_id * TILE_SIZE_K + offs_edge
    offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
    par_start = tl.load(pids_start + offs_estart)
    epars_ptr = params + \
        offs_node[:,None] + \
        par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `element_mars` and `element_ents`
    edge_start = tl.load(cids_start + offs_estart)
    emars_ptr = element_mars + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    eents_ptr = element_ents + \
        edge_start[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
    cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

    # Node mars
    off_nids = tl.load(nids + nblock_id)
    offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
    nmars = tl.load(node_mars + offs_nmars, mask = mask_batch[None,:])
    nentgrads = tl.load(node_ent_grads + offs_nmars, mask = mask_batch[None,:])

    # Inner loop
    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        eents = tl.load(eents_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        # epars * exp(emars) * eents / exp(nmars)
        val1 = emars + tl.log(eents) # [TILE_SIZE_K, BLOCK_B]
        val1_max = tl.max(val1, axis = 0)[None,:]
        val1_sub = tl.where(val1_max != -float("inf"), tl.exp(val1 - val1_max), 0.0)

        epars_bf16 = epars.to(tl.float32)
        val1_bf16 = val1_sub.to(tl.float32)
        out1 = tl.dot(epars_bf16, val1_bf16).to(tl.float32)
        acc += out1 * tl.exp(val1_max - nmars)

        # -epars * log(epars) * exp(emars) / exp(nmars)
        val2 = epars * (tl.log(epars) + 1.0)
        emars_max = tl.max(emars, axis = 0)[None,:]
        emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

        val2_bf16 = val2.to(tl.float32)
        emars_bf16 = emars_sub.to(tl.float32)
        out2 = tl.dot(val2_bf16, emars_bf16).to(tl.float32)
        acc -= out2 * tl.exp(emars_max - nmars)

        # -epars * emars * exp(emars) / exp(nmars)
        val3_sub = emars_sub * emars
        out3 = tl.dot(epars, val3_sub)
        acc -= out3 * tl.exp(emars_max - nmars)

        # epars * nmars * exp(nmars) / exp(nmars)
        out4 = tl.dot(epars_bf16, emars_bf16).to(tl.float32)
        acc += out4 * tl.exp(emars_max - nmars) * nmars

        # Increment `epars_ptr`
        pids_inc = tl.load(pids_inc_ptr)
        epars_ptr += pids_inc[None,:]
        pids_inc_ptr += TILE_SIZE_K

        # Increment `emars_ptr` and `eents_ptr`
        cids_inc = tl.load(cids_inc_ptr)
        emars_ptr += cids_inc[:,None] * batch_size
        eents_ptr += cids_inc[:,None] * batch_size
        cids_inc_ptr += TILE_SIZE_K

    tl.atomic_add(node_flows + offs_nmars, acc * nentgrads * lamda, mask = mask_batch[None,:])


def sum_layer_cond_ent_grad_params_triton2(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda):

    params = pc.params
    param_flows = pc.param_flows
    node_mars = pc.node_mars
    element_mars = pc.element_mars
    node_flows = pc.node_flows
    element_flows = pc.element_flows
    
    for partition_id in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[partition_id]
        cids = layer.partitioned_cids[partition_id]
        pids = layer.partitioned_pids[partition_id]
        pfids = layer.partitioned_pfids[partition_id]

        block_size = layer.block_size
        num_nblocks = nids.size(0)
        layer_n_nodes = num_nblocks * block_size
        num_edges = cids.size(1)
        batch_size = node_ents.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        base_size = min(block_size, num_edges, BATCH_SIZE_NP2, 128)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(2048 // remainder, base_size * remainder, num_edges)
        TILE_SIZE_M = min(2048 // TILE_SIZE_K, block_size)
        BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        signature = ("block_sparse", partition_id, TILE_SIZE_K)
        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16 and signature in layer._cached_fw_pcids:
            cids_start, cids_increment, pids_start, pids_increment = layer._cached_fw_pcids[signature]

            BLOCK_SIZE_M = block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            sum_layer_cond_ent_grad_params_kernel2[grid](
                node_ents, 
                element_ents, 
                node_mars,
                element_mars,
                node_flows, 
                node_ent_grads,
                params, 
                nids, 
                cids_start,
                cids_increment, 
                pids_start,
                pids_increment,
                batch_size,
                lamda,
                BLOCK_B = BLOCK_B,
                TILE_SIZE_K = TILE_SIZE_K,
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M,
                BLOCK_SIZE_M = BLOCK_SIZE_M
            )

        elif block_size == 1 and nids.size(0) == 1:
            nmars = pc.node_mars[nids[0],:]
            nentgrads = node_ent_grads[nids[0],:]
            
            params = pc.params[pids[0,:]]
            cmars = pc.element_mars[cids[0,:],:]
            cents = element_ents[cids[0,:],:]

            cond_params = cmars + params[:,None].log() - nmars[None,:]
            ngrads = (nentgrads[None,:] * cond_params.exp() * (cents - cond_params.clamp(min = -1000.0) - 1)).sum(dim = 0)

            node_flows[nids[0],:] += ngrads * lamda

        else:
            raise NotImplementedError()


@triton.jit
def sum_layer_cond_ent_grad_params_kernel3(node_flows, element_flows, node_mars, element_mars, node_ents, element_ents, 
                                           node_ent_grads, element_ent_grads, params, chids, parids_start, parids_increment,
                                           parpids_start, parpids_increment, lamda, batch_size: tl.constexpr, ptr_inc_step: tl.constexpr,
                                           BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                           TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = params + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars`
    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    nentgrads_ptr = node_ent_grads + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Initialize pointers to `element_mars` (only when using MPE propagation method)
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    # Initialize pointers to `element_mars` (only when using MPE propagation method)
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]
    eents_ptr = element_ents + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    eents = tl.load(eents_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        nentgrads = tl.load(nentgrads_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        # nentgrads * epars * exp(emars) * eents / exp(nmars)
        val1 = emars + tl.log(eents) # [TILE_SIZE_M, BLOCK_B]
        emars_max = tl.max(emars, axis = 0) # [BLOCK_B]
        val2_sub = nentgrads * tl.exp(emars_max[None,:] - nmars) # [TILE_SIZE_K, BLOCK_B]

        epars_bf16 = epars.to(tl.float32)
        out1 = tl.dot(epars_bf16, val2_sub).to(tl.float32)
        acc += out1 * tl.exp(val1 - emars_max[None,:])

        # -epars * log(epars) * exp(emars) / exp(nmars)
        val3 = epars * (tl.log(epars) + 1.0) # [TILE_SIZE_M, TILE_SIZE_K]
        nmars_sub = nentgrads * tl.exp(emars_max[None,:] - nmars) # [TILE_SIZE_K, BLOCK_B]

        val3_bf16 = val3.to(tl.float32)
        out2 = tl.dot(val3_bf16, nmars_sub).to(tl.float32)
        acc -= out2 * tl.exp(emars - emars_max[None,:])

        # -epars * emars * exp(emars) / exp(nmars)
        out3 = tl.dot(epars, nmars_sub).to(tl.float32)
        acc -= out3 * tl.exp(emars - emars_max[None,:]) * emars

        # epars * nmars * exp(emars) / exp(nmars)
        val4 = nentgrads * tl.exp(emars_max[None,:] - nmars) * nmars # [TILE_SIZE_K, BLOCK_B]
        out4 = tl.dot(epars, val4).to(tl.float32)
        acc += out4 * tl.exp(emars - emars_max[None,:])

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[:,None] * batch_size
        nentgrads_ptr += parids_inc[:,None] * batch_size
        parids_inc_ptr += ptr_inc_step

    acc = -acc * lamda

    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    tl.atomic_add(element_flows + offs_elemfs, acc, mask = mask_batch[None,:])


def sum_layer_cond_ent_grad_params_triton3(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda):

    params = pc.params
    param_flows = pc.param_flows
    node_mars = pc.node_mars
    element_mars = pc.element_mars
    node_flows = pc.node_flows
    element_flows = pc.element_flows

    for partition_id in range(layer.num_bk_partitions):
        chids = layer.partitioned_chids[partition_id]
        parids = layer.partitioned_parids[partition_id]
        parpids = layer.partitioned_parpids[partition_id]
        cs_block_size = layer.cs_block_sizes[partition_id]

        num_nblocks = chids.size(0)
        layer_n_nodes = num_nblocks * cs_block_size
        block_size = layer.block_size
        num_edges = parids.size(1) * block_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(block_size, num_edges, BATCH_SIZE_NP2, 64)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(512, base_size * remainder, num_edges)
        TILE_SIZE_M = min(2048 // TILE_SIZE_K, cs_block_size)
        BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
            signature = ("block_sparse", partition_id, TILE_SIZE_K)
            if signature not in layer._cached_bk_parids:
                # Pre-compute pointer increments for `parids` and `parpids`

                if TILE_SIZE_K < layer.block_size:
                    ptr_inc_step = 1

                    num_rep = layer.block_size // TILE_SIZE_K
                    parids = (parids[:,:,None].repeat(1, 1, num_rep) + \
                        torch.arange(0, layer.block_size, TILE_SIZE_K, device = parids.device)[None,None,:]).reshape(
                            parids.size(0), K_NUM_TILES, 1)
                    parpids = (parpids[:,:,None].repeat(1, 1, num_rep) + \
                        torch.arange(0, layer.block_size, TILE_SIZE_K, device = parpids.device)[None,None,:]).reshape(
                            parpids.size(0), K_NUM_TILES, 1)

                else:
                    ptr_inc_step = TILE_SIZE_K // layer.block_size

                    parids = parids.reshape(parids.size(0), K_NUM_TILES, ptr_inc_step)
                    parpids = parpids.reshape(parpids.size(0), K_NUM_TILES, ptr_inc_step)

                parids_start = parids[:,0,:].contiguous()
                parids_increment = torch.cat(
                    (parids[:,1:,:] - parids[:,:-1,:], parids[:,0:1,:] * 0),
                    dim = 1
                ).contiguous()

                parpids_start = parpids[:,0,:].contiguous()
                parpids_increment = torch.cat(
                    (parpids[:,1:,:] - parpids[:,:-1,:], parpids[:,0:1,:] * 0),
                    dim = 1
                ).contiguous()

                layer._cached_bk_parids[signature] = [parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step]
            else:
                parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step = layer._cached_bk_parids[signature]

            BLOCK_SIZE_M = cs_block_size
            BLOCK_SIZE_K = block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            sum_layer_cond_ent_grad_params_kernel3[grid](
                node_flows = node_flows, 
                element_flows = element_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                node_ents = node_ents, 
                element_ents = element_ents, 
                node_ent_grads = node_ent_grads, 
                element_ent_grads = element_ent_grads,
                params = params, 
                chids = chids, 
                parids_start = parids_start,
                parids_increment = parids_increment,
                parpids_start = parpids_start,
                parpids_increment = parpids_increment, 
                lamda = lamda,
                batch_size = batch_size, 
                ptr_inc_step = ptr_inc_step,
                BLOCK_B = BLOCK_B, 
                TILE_SIZE_K = TILE_SIZE_K, 
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = BLOCK_SIZE_M,
                BLOCK_SIZE_K = BLOCK_SIZE_K
            )

        elif block_size == 1 and parids.size(1) * block_size == 1:
            assert layer.num_fw_partitions == 1

            nids = layer.partitioned_nids[0]
            cids = layer.partitioned_cids[0]
            pids = layer.partitioned_pids[0]

            nmars = pc.node_mars[nids[0],:]
            nentgrads = node_ent_grads[nids[0],:]
            
            params = pc.params[pids[0,:]]
            cmars = pc.element_mars[cids[0,:],:]
            cents = element_ents[cids[0,:],:]

            cond_params = cmars + params[:,None].log() - nmars[None,:]
            cgrads = (nentgrads[None,:] * cond_params.exp() * (cents - cond_params - 1))

            element_flows[cids[0,:],:] -= cgrads * lamda

        else:
            raise NotImplementedError()


@torch.compile
def update_fn3(cmars, params, nmars, nentgrads, element_ent_grads, chids, n_blk_params, block_size, ch_block_size, i, j):
    cond_params = cmars[:,None,:] + params[:,:,None].log() - nmars[None,:,:]

    centgrads = (nentgrads[None,:,:] * cond_params.exp()).sum(dim = 1)
    element_ent_grads[chids[i]:chids[i]+ch_block_size,:] = centgrads


def sum_layer_cond_ent_grad_chs(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads):

    block_size = layer.block_size
    for partition_id in range(layer.num_bk_partitions):
        chids = layer.partitioned_chids[partition_id]
        parids = layer.partitioned_parids[partition_id]
        parpids = layer.partitioned_parpids[partition_id]
        ch_block_size = layer.cs_block_sizes[partition_id]

        n_blk_params = block_size * ch_block_size
        for i in range(chids.size(0)):
            cmars = pc.element_mars[chids[i]:chids[i]+ch_block_size,:]
            for j in range(0, parids.size(1)):
                params = pc.params[parpids[i,j]:parpids[i,j]+n_blk_params].reshape(ch_block_size, block_size)
                nmars = pc.node_mars[parids[i,j]:parids[i,j]+block_size,:]
                nentgrads = node_ent_grads[parids[i,j]:parids[i,j]+block_size,:]

                update_fn3(cmars, params, nmars, nentgrads, element_ent_grads, chids, n_blk_params, block_size, ch_block_size, i, j)


@triton.jit
def sum_layer_cond_ent_grad_chs_kernel(node_flows, element_flows, node_mars, element_mars, node_ents, element_ents, 
                                       node_ent_grads, element_ent_grads, params, chids, parids_start, parids_increment,
                                       parpids_start, parpids_increment, batch_size: tl.constexpr, ptr_inc_step: tl.constexpr,
                                       BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                       TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
    pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

    # Get inferred node block id from `pid_m`
    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    # Initialize pointers to `params`
    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_edge = tl.arange(0, TILE_SIZE_K)
    offs_edge_gid = offs_edge // BLOCK_SIZE_K
    offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
    par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    epars_ptr = params + \
        offs_ele[:,None] * BLOCK_SIZE_K + \
        (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

    # Batch offsets and mask
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    # Initialize pointers to `node_mars`
    edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
    nmars_ptr = node_mars + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
    nentgrads_ptr = node_ent_grads + \
        (edge_start + offs_edge_nid)[:,None] * batch_size + \
        offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

    # Batch increment pointers
    parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
    parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

    # Initialize pointers to `element_mars` (only when using MPE propagation method)
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    # Initialize pointers to `element_mars` (only when using MPE propagation method)
    off_eleids = tl.load(chids + eleblock_id)
    emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]
    eents_ptr = element_ents + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    eents = tl.load(eents_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, BLOCK_B]

    acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

    for k in range(0, K_NUM_TILES):
        epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
        nentgrads = tl.load(nentgrads_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        # nentgrads * epars * exp(emars) / exp(nmars)
        val1 = emars # [TILE_SIZE_M, BLOCK_B]
        emars_max = tl.max(emars, axis = 0) # [BLOCK_B]
        val2_sub = nentgrads * tl.exp(emars_max[None,:] - nmars) # [TILE_SIZE_K, BLOCK_B]

        epars_bf16 = epars.to(tl.float32)
        out1 = tl.dot(epars_bf16, val2_sub).to(tl.float32)
        acc += out1 * tl.exp(val1 - emars_max[None,:])

        # Increment `epars_ptr`
        parpids_inc = tl.load(parpids_inc_ptr)
        epars_ptr += parpids_inc[None,:]
        parpids_inc_ptr += ptr_inc_step

        # Increment `nmars_ptr`
        parids_inc = tl.load(parids_inc_ptr)
        nmars_ptr += parids_inc[:,None] * batch_size
        nentgrads_ptr += parids_inc[:,None] * batch_size
        parids_inc_ptr += ptr_inc_step

    offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
    tl.store(element_ent_grads + offs_elemfs, acc, mask = mask_batch[None,:])


def sum_layer_cond_ent_grad_chs_triton(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads):

    params = pc.params
    param_flows = pc.param_flows
    node_mars = pc.node_mars
    element_mars = pc.element_mars
    node_flows = pc.node_flows
    element_flows = pc.element_flows

    for partition_id in range(layer.num_bk_partitions):
        chids = layer.partitioned_chids[partition_id]
        parids = layer.partitioned_parids[partition_id]
        parpids = layer.partitioned_parpids[partition_id]
        cs_block_size = layer.cs_block_sizes[partition_id]

        num_nblocks = chids.size(0)
        layer_n_nodes = num_nblocks * cs_block_size
        block_size = layer.block_size
        num_edges = parids.size(1) * block_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(block_size, num_edges, BATCH_SIZE_NP2, 64)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(512, base_size * remainder, num_edges)
        TILE_SIZE_M = min(2048 // TILE_SIZE_K, cs_block_size)
        BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
            signature = ("block_sparse", partition_id, TILE_SIZE_K)
            parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step = layer._cached_bk_parids[signature]

            BLOCK_SIZE_M = cs_block_size
            BLOCK_SIZE_K = block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            sum_layer_cond_ent_grad_chs_kernel[grid](
                node_flows = node_flows, 
                element_flows = element_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                node_ents = node_ents, 
                element_ents = element_ents, 
                node_ent_grads = node_ent_grads, 
                element_ent_grads = element_ent_grads,
                params = params, 
                chids = chids, 
                parids_start = parids_start,
                parids_increment = parids_increment,
                parpids_start = parpids_start,
                parpids_increment = parpids_increment, 
                batch_size = batch_size, 
                ptr_inc_step = ptr_inc_step,
                BLOCK_B = BLOCK_B, 
                TILE_SIZE_K = TILE_SIZE_K, 
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = BLOCK_SIZE_M,
                BLOCK_SIZE_K = BLOCK_SIZE_K
            )

        elif block_size == 1 and parids.size(1) * block_size == 1:
            assert layer.num_fw_partitions == 1

            nids = layer.partitioned_nids[0]
            cids = layer.partitioned_cids[0]
            pids = layer.partitioned_pids[0]

            nmars = pc.node_mars[nids[0],:]
            nentgrads = node_ent_grads[nids[0],:]
            
            params = pc.params[pids[0,:]]
            cmars = pc.element_mars[cids[0,:],:]
            cents = element_ents[cids[0,:],:]

            cond_params = cmars + params[:,None].log() - nmars[None,:]
            cgrads = nentgrads[None,:] * cond_params.exp()

            element_ent_grads[cids[0,:],:] += cgrads

        else:
            raise NotImplementedError()


def compute_conditional_ent_bp(pc, x, node_ents, lamda):

    with torch.no_grad():

        B = x.size(0)

        # Initialize buffers for conditional ent backward pass
        node_ent_grads = torch.zeros_like(pc.node_mars)
        element_ent_grads = torch.zeros_like(pc.element_mars)
        element_ents = torch.zeros_like(pc.element_mars)

        pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, B), set_value = 0.0)
        pc._init_buffer(name = "element_flows", shape = (pc.num_elements, B), set_value = 0.0)

        node_ent_grads[pc._root_node_range[0]:pc._root_node_range[1],:] = 1.0
        pc.node_flows[pc._root_node_range[0]:pc._root_node_range[1],:] = 1.0

        # The CondENT back pass
        for layer_id in range(len(pc.inner_layer_groups) - 1, -1, -1):
            layer_group = pc.inner_layer_groups[layer_id]

            if layer_group.is_prod():
                layer_group.backward(node_ent_grads, element_ent_grads, logspace_flows = False)
                layer_group.backward(pc.node_flows, pc.element_flows, logspace_flows = False)

            elif layer_group.is_sum():
                # Sum layer

                prod_layer = pc.inner_layer_groups[layer_id-1]
                prod_layer.forward(node_ents, element_ents, _for_backward = True)
                prod_layer.forward(pc.node_mars, pc.element_mars, _for_backward = True)

                pc.element_flows[:,:] = 0.0

                for layer in layer_group:
                    # sum_layer_cond_ent_grad_params(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda)
                    sum_layer_cond_ent_grad_params_triton1(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda)
                    sum_layer_cond_ent_grad_params_triton2(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda)
                    sum_layer_cond_ent_grad_params_triton3(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads, lamda)

                    # sum_layer_cond_ent_grad_chs(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads)
                    sum_layer_cond_ent_grad_chs_triton(layer, pc, node_ents, element_ents, node_ent_grads, element_ent_grads)

                layer_group.backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params, 
                                     param_flows = pc.param_flows, allow_modify_flows = False, 
                                     propagation_alg = "LL", logspace_flows = False, negate_pflows = False, 
                                     accumulate_ch_flows = True, allow_neg_flows = True)

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")