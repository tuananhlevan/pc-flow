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
from src.sgd.conditional_ent import compute_conditional_ent, compute_conditional_ent_bp


@triton.jit
def _accum_flows_all_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                            metadata_ptr, s_mids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                            batch_size: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                            BLOCK_SIZE: tl.constexpr, C: tl.constexpr, num_cats: tl.constexpr, neg: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_nodes * batch_size

    # Raw batch and (local) node id
    batch_offsets = (offsets % batch_size)
    local_offsets = (offsets // batch_size)

    if partial_eval > 0:
        local_offsets = tl.load(bk_local_ids_ptr + local_offsets, mask = mask, other = 0)

    s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
    s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)

    ns_offsets = (local_offsets + node_offset) * batch_size + batch_offsets
    flows = tl.load(node_flows_ptr + ns_offsets, mask = mask, other = 0)

    if logspace_flows:
        flows = tl.exp(flows)

    pf_offsets = s_pfids[:,None] + tl.arange(0, num_cats)[None,:]
    p_offsets = s_pids[:,None] + tl.arange(0, num_cats)[None,:]

    params = tl.load(params_ptr + p_offsets, mask = (mask[:,None] & (tl.arange(0, num_cats) < num_cats)[None,:]), other = 0.0)
    cum_params = tl.sum(params, axis = 1)

    sflows = flows[:,None] * (params / (cum_params[:,None] + 1e-12))

    if neg:
        tl.atomic_add(param_flows_ptr + pf_offsets, -1.0 * sflows, mask = (mask[:,None] & (tl.arange(0, num_cats) < num_cats)[None,:]))
    else:
        tl.atomic_add(param_flows_ptr + pf_offsets, sflows, mask = (mask[:,None] & (tl.arange(0, num_cats) < num_cats)[None,:]))


def accum_flows_all(self, node_flows: torch.Tensor, node_mars: torch.Tensor, logspace_flows: bool = False,
                    neg: bool = True):
    """
    data: [num_vars, B]
    node_flows: [num_nodes, B]
    node_mars: [num_nodes, B]
    """

    params = self.params

    assert params.dim() == 1

    tot_num_nodes = node_flows.size(0)
    batch_size = node_flows.size(1)
    node_offset = self._output_ind_range[0]

    if not self.provided("bk_local_ids"):
        layer_num_nodes = self._output_ind_range[1] - self._output_ind_range[0]
        bk_local_ids = None
    else:
        layer_num_nodes = self.bk_local_ids.size(0)
        bk_local_ids = self.bk_local_ids

    BLOCK_SIZE = 128

    C = triton.next_power_of_2(self.nodes[0].dist.num_cats)

    grid = (triton.cdiv(layer_num_nodes * batch_size, BLOCK_SIZE),)

    _accum_flows_all_kernel[grid](
        params_ptr = self.params,
        param_flows_ptr = self.param_flows,
        node_flows_ptr = node_flows, 
        node_mars_ptr = node_mars,
        vids_ptr = self.vids, 
        s_pids_ptr = self.s_pids,
        s_pfids_ptr = self.s_pfids,
        metadata_ptr = self.metadata, 
        s_mids_ptr = self.s_mids, 
        bk_local_ids_ptr = bk_local_ids,
        layer_num_nodes = layer_num_nodes, 
        batch_size = batch_size,
        nv_block_size = triton.next_power_of_2(self.num_vars_per_node),
        node_offset = node_offset, 
        BLOCK_SIZE = BLOCK_SIZE, 
        partial_eval = 1 if bk_local_ids is not None else 0,
        logspace_flows = logspace_flows,
        C = C,
        num_cats = self.nodes[0].dist.num_cats,
        neg = neg,
        num_warps = 8
    )


@triton.jit
def acc_all_kernel(node_flows_ptr, pfids_ptr, pids_ptr, params_ptr, param_flows_ptr, nsid: tl.constexpr, num_cats: tl.constexpr, 
                   batch_size: tl.constexpr, neg: tl.constexpr, layer_num_nodes: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_C: tl.constexpr,
                   norm_params: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_nodes

    n_pfids = tl.load(pfids_ptr + offsets, mask = mask)
    n_pids = tl.load(pids_ptr + offsets, mask = mask)

    nflows = tl.load(node_flows_ptr + (nsid + offsets) * batch_size, mask = mask)

    coffs = tl.arange(0, BLOCK_C)

    pfids = n_pfids[:,None] + coffs[None,:]
    pids = n_pids[:,None] + coffs[None,:]

    cmask = coffs < num_cats

    params = tl.load(params_ptr + pids, mask = mask[:,None])
    if norm_params:
        norm_param = params / (tl.sum(params, axis = 1)[:,None] + 1e-12)
    else:
        norm_param = params

    if neg:
        tl.atomic_add(param_flows_ptr + pfids, tl.exp(nflows[:,None]) * norm_param * -batch_size, mask = (mask[:,None] & cmask[None,:]))
    else:
        tl.atomic_add(param_flows_ptr + pfids, tl.exp(nflows[:,None]) * norm_param * batch_size, mask = (mask[:,None] & cmask[None,:]))


@triton.jit
def acc_all_kernel_large(node_flows_ptr, pfids_ptr, pids_ptr, params_ptr, param_flows_ptr, nsid: tl.constexpr, num_cats: tl.constexpr, 
                         batch_size: tl.constexpr, neg: tl.constexpr, layer_num_nodes: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                         BLOCK_C: tl.constexpr, nc: tl.constexpr, norm_params: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_nodes

    n_pfids = tl.load(pfids_ptr + offsets, mask = mask)
    n_pids = tl.load(pids_ptr + offsets, mask = mask)

    nflows = tl.load(node_flows_ptr + (nsid + offsets) * batch_size, mask = mask)

    coffs1 = tl.arange(0, BLOCK_C)
    cum_params = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

    for i in range(nc):
        pfids = n_pfids[:,None] + coffs1[None,:]
        pids = n_pids[:,None] + coffs1[None,:]

        cmask = coffs1 < num_cats

        params = tl.load(params_ptr + pids, mask = (mask[:,None] & cmask[None,:]), other = 0.0)

        cum_params += tl.sum(params, axis = 1)

        coffs1 += BLOCK_C

    coffs2 = tl.arange(0, BLOCK_C)

    for i in range(nc):
        pfids = n_pfids[:,None] + coffs2[None,:]
        pids = n_pids[:,None] + coffs2[None,:]

        cmask = coffs2 < num_cats

        params = tl.load(params_ptr + pids, mask = (mask[:,None] & cmask[None,:]), other = 0.0)
        if norm_params:
            norm_param = params / (cum_params[:,None] + 1e-12)
        else:
            norm_param = params

        if neg:
            tl.atomic_add(param_flows_ptr + pfids, tl.exp(nflows[:,None]) * norm_param * -batch_size, mask = (mask[:,None] & cmask[None,:]))
        else:
            tl.atomic_add(param_flows_ptr + pfids, tl.exp(nflows[:,None]) * norm_param * batch_size, mask = (mask[:,None] & cmask[None,:]))

        coffs2 += BLOCK_C


@triton.jit
def _sgd_kernel(params_ptr, param_flows_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr,
                source_nids_ptr, constexprs_ptr, layer_num_source_nodes: tl.constexpr, 
                BLOCK_SIZE: tl.constexpr, update_log_params: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    lr = tl.load(constexprs_ptr)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < layer_num_source_nodes

    # Get the local node ids
    local_offsets = tl.load(source_nids_ptr + offsets, mask = mask, other = 0)

    # Get the corresponding start id for `params` and `param_flows`
    s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
    s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)

    # Get `num_cats` from `metadata`
    s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
    num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

    max_num_cats = tl.max(num_cats, axis = 0)

    # Parameter update
    for cat_id in range(max_num_cats):
        cat_mask = mask & (cat_id < num_cats)

        param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
        flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

        if update_log_params:
            new_param = tl.exp(tl.log(param) + lr * flow)
        else:
            w = param + tlmath.log1p(-tl.exp(-param))
            wgrad = flow / (param + 1e-12) * tl.where(w > 0, 1.0 / (1.0 + tl.exp(-w)), tl.exp(w) / (tl.exp(w) + 1.0))
            w = tl.where(w > -float("inf"), w + lr * wgrad, -float("inf"))

            new_param = tl.where(w < 0, tlmath.log1p(tl.exp(w)), w + tlmath.log1p(tl.exp(-w)))

        tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)


def sgd_input_layer(self, lr: float, update_log_params: bool = True):
    # Normalize and update parameters
    with torch.no_grad():

        # Accumulate parameter flows of tied nodes
        for i in range(len(self.tied2source_nids)):
            pfid_start, num_par_flows, ch_pfids = self.tied2source_nids[i]
            num_coalesced_blocks = ch_pfids.size(0)

            if num_coalesced_blocks <= 1024:
                BLOCK_N = triton.next_power_of_2(num_coalesced_blocks)
                BLOCK_M = min(1024 // BLOCK_N, num_par_flows)

                grid = (triton.cdiv(num_par_flows, BLOCK_M),)

                self._pflow_accum_kernel[grid](
                    param_flows_ptr = self.param_flows,
                    pfid_start = pfid_start,
                    ch_pfids_ptr = ch_pfids,
                    num_coalesced_blocks = num_coalesced_blocks,
                    num_par_flows = num_par_flows,
                    BLOCK_M = BLOCK_M,
                    BLOCK_N = BLOCK_N
                )
            else:
                raise NotImplementedError("Unsupported number of coalesced parameter flows.")


        layer_num_source_nodes = self.source_nids.size(0)

        constexprs = torch.tensor([lr], dtype = torch.float32, device = self.device)

        BLOCK_SIZE = 1024

        grid = (triton.cdiv(layer_num_source_nodes, BLOCK_SIZE),)

        _sgd_kernel[grid](
            params_ptr = self.params,
            param_flows_ptr = self.param_flows,
            s_pids_ptr = self.s_pids,
            s_pfids_ptr = self.s_pfids,
            metadata_ptr = self.metadata,
            s_mids_ptr = self.s_mids,
            source_nids_ptr = self.source_nids,
            constexprs_ptr = constexprs,
            layer_num_source_nodes = layer_num_source_nodes,
            BLOCK_SIZE = BLOCK_SIZE,
            update_log_params = update_log_params,
            num_warps = 8
        )


def momentum_update(gradients, m, t, momentum, dampening, nesterov = False):
    with torch.no_grad():
        for idx, (grad, m_t) in enumerate(zip(gradients, m)):
            if m_t is None:
                m_t = torch.zeros(grad.size(), device = grad.device)
                m[idx] = m_t
            if t > 1:
                m_t[:] = momentum * m_t[:] + (1.0 - dampening) * grad
            else:
                m_t[:] = grad
            if nesterov:
                grad[:] = grad + momentum * m_t
            else:
                grad[:] = m_t


def adam_update(gradients, m, v, t, beta1=0.9, beta2=0.95, epsilon=1e-8):
    with torch.no_grad():
        for idx, (grad, m_t, v_t) in enumerate(zip(gradients, m, v)):
            if m_t is None:
                m_t = torch.zeros(grad.size(), device = grad.device)
                v_t = torch.zeros(grad.size(), device = grad.device)
                m[idx] = m_t
                v[idx] = v_t
            m_t[:] = beta1 * m_t + (1 - beta1) * grad
            v_t[:] = beta2 * v_t + (1 - beta2) * (grad ** 2)
            m_t_hat = m_t / (1 - beta1 ** t)
            v_t_hat = v_t / (1 - beta2 ** t)
            grad[:] = (m_t_hat / (torch.sqrt(v_t_hat) + epsilon))


def adam_update_with_decay(parameters, gradients, m, v, t, w_decay=1e-5, beta1=0.9, beta2=0.999, epsilon=1e-8):
    with torch.no_grad():
        for idx, (param, grad, m_t, v_t) in enumerate(zip(parameters, gradients, m, v)):
            if m_t is None:
                m_t = torch.zeros(grad.size(), device = grad.device)
                v_t = torch.zeros(grad.size(), device = grad.device)
                m[idx] = m_t
                v[idx] = v_t

            # Misalignment between params and flows......
            if param.size(0) == grad.size(0):
                pass
            elif (math.log(param.size(0) - grad.size(0))/math.log(2)).is_integer():
                c = param.size(0) - grad.size(0)
                param = param[c:]
            else:
                raise NotImplementedError("Case not handled.")

            grad -= w_decay * (param > math.e).float() * (param + 1e-8).log()
            
            m_t[:] = beta1 * m_t + (1 - beta1) * grad
            v_t[:] = beta2 * v_t + (1 - beta2) * (grad ** 2)
            m_t_hat = m_t / (1 - beta1 ** t)
            v_t_hat = v_t / (1 - beta2 ** t)
            grad[:] = (m_t_hat / (torch.sqrt(v_t_hat) + epsilon))


def adam_softplus_update(parameters, gradients, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    with torch.no_grad():
        for idx, (param, grad, m_t, v_t) in enumerate(zip(parameters, gradients, m, v)):
            if m_t is None:
                m_t = torch.zeros(grad.size(), device = grad.device)
                v_t = torch.zeros(grad.size(), device = grad.device)
                m[idx] = m_t
                v[idx] = v_t

            # Misalignment between params and flows......
            if param.size(0) == grad.size(0):
                pass
            elif (math.log(param.size(0) - grad.size(0))/math.log(2)).is_integer():
                c = param.size(0) - grad.size(0)
                param = param[c:]
            else:
                raise NotImplementedError("Case not handled.")

            w = param + torch.log1p(-torch.exp(-param))
            factor = torch.where(w == -float("inf"), 
                0.0,
                torch.where(w > 0, 1.0 / (1.0 + torch.exp(-w)), torch.exp(w) / (torch.exp(w) + 1.0)) / (param + 1e-12)
            )
            wgrad = grad * factor

            m_t[:] = beta1 * m_t + (1 - beta1) * wgrad
            v_t[:] = beta2 * v_t + (1 - beta2) * (wgrad ** 2)
            m_t_hat = m_t / (1 - beta1 ** t)
            v_t_hat = v_t / (1 - beta2 ** t)

            scaled_wgrad = (m_t_hat / (torch.sqrt(v_t_hat) + epsilon))

            grad[:] = scaled_wgrad / (factor + 1e-12)


@triton.jit
def cum_param_kernel(cum_params, params, param_flows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, 
                     global_nids, constexprs, num_blocks, BLOCK_ID: tl.constexpr, 
                     BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)

    # Retrieve the constants
    pseudocount = tl.load(constexprs + 1)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_param = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_param = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    param = tl.load(params + offs_param, mask = mask_param, other = 0)

    nparam = tl.sum(param, axis = 1)

    tl.atomic_add(cum_params + global_nid, nparam, mask = mask_m)


@triton.jit
def sub_pars_kernel(params, param_flows, cum_params, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals,
                         global_nids, constexprs, num_blocks, BLOCK_ID: tl.constexpr, 
                         BLOCK_SIZE: tl.constexpr, w):

    pid = tl.program_id(axis = 0)

    # Retrieve the constants
    step_size = tl.load(constexprs)
    pseudocount = tl.load(constexprs + 1)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    pflow_start = tl.load(pflow_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_pflow = pflow_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_pflow = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    # pflows = tl.load(param_flows + offs_pflow, mask = mask_pflow, other = 0)

    nparam = tl.load(cum_pflows + global_nid, mask = mask_m, other = 1)

    offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    old_param = tl.load(params + offs_par, mask = mask_pflow, other = 0)

    tl.atomic_add(param_flows + offs_pflow, -(old_param / nparam[:,None]) * w, mask = mask_pflow)


def sub_partition_pflows(params: torch.Tensor, param_flows: torch.Tensor, par_update_kwargs, w: float):

    par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = par_update_kwargs

    tot_num_nodes = metadata["tot_num_nodes"]
    BLOCK_SIZE = metadata["BLOCK_SIZE"]

    if cum_pflows is None:
        cum_pflows = torch.zeros([tot_num_nodes], dtype = torch.float32, device = params.device)
    else:
        cum_pflows[:] = 0.0

    num_blocks = par_start_ids.size(0)
    BLOCK_ID = 2048 // BLOCK_SIZE

    grid = (triton.cdiv(num_blocks, BLOCK_ID),)

    constexprs = torch.tensor([step_size, pseudocount]).to(params.device)

    keep_zero_params = 1 if keep_zero_params else 0

    cum_param_kernel[grid](
        cum_pflows, params, param_flows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, 
        global_nids, constexprs, num_blocks, BLOCK_ID, BLOCK_SIZE
    )

    sub_pars_kernel[grid](
        params, param_flows, cum_pflows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals,
        global_nids, constexprs, num_blocks, BLOCK_ID, BLOCK_SIZE, w
    )

    return None


class SGDWrapper():

    DEFAULT_DIV_FACTOR = 100.0

    def __init__(self, pc, mode = "partition", optimizer = "Adam", update_log_params = True, w_decay = 0.0, 
                 input_layer_norm_params: bool = True):
        self.pc = pc

        self.mode = mode
        self.optimizer = optimizer

        self.t = 0

        self.m = None
        self.v = None

        self.momentum = 0.9
        self.dampening = 0.0
        self.nesterov = False

        self.samples_consumed = 0

        self.update_log_params = update_log_params

        self.w_decay = w_decay

        self.input_layer_norm_params = input_layer_norm_params

    def optim_step(self, inputs: torch.Tensor, lr: float, update: bool, no_param_clipping: bool = False, **kwargs):

        self.t += 1

        if self.mode == "ll":
            pass

        elif self.mode == "partition":
            # log f(x) - log Z

            ## The first pass ##
            lls = self.pc(inputs, propagation_alg = "LL", force_use_fp32 = False)
            self.pc.backward(inputs, flows_memory = 1.0, allow_modify_flows = False,
                            propagation_alg = "LL", logspace_flows = True)

            # pflows1 = self.pc.param_flows.detach().cpu().clone()
            # piflows1 = self.pc.input_layer_group[0].param_flows.detach().cpu().clone()

            ## The second pass ##
            self.partition_eval(negate_pflows = True)

            # pflows2 = pflows1 - self.pc.param_flows.detach().cpu().clone()
            # piflows2 = piflows1 - self.pc.input_layer_group[0].param_flows.detach().cpu().clone()

            # import pdb; pdb.set_trace()

        elif self.mode.startswith("partition_lambda_"):
            lamda = float(self.mode[17:])

            ## The first pass ##
            lls = self.pc(inputs, propagation_alg = "LL", force_use_fp32 = False)
            self.pc.backward(inputs, flows_memory = 1.0, allow_modify_flows = False,
                            propagation_alg = "LL", logspace_flows = True)

            # Scale gradients
            self.pc.param_flows /= lamda
            for layer in self.pc.input_layer_group:
                layer.param_flows /= lamda

            ## The second pass ##
            self.partition_eval(negate_pflows = True)

            # Scale gradients
            self.pc.param_flows *= lamda
            for layer in self.pc.input_layer_group:
                layer.param_flows *= lamda

        elif self.mode.startswith("ent_reg_"):
            lamda = float(self.mode[8:])

            ## The first pass ##
            lls, ents, node_ents = compute_conditional_ent(self.pc, inputs, ret_node_ents = True)
            self.aveg_ent = ents.mean().item()

            compute_conditional_ent_bp(self.pc, inputs, node_ents, lamda)

            # Compute backward pass for all input layers
            for idx, layer in enumerate(self.pc.input_layer_group):
                layer.backward(inputs.permute(1, 0).contiguous(), self.pc.node_flows, self.pc.node_mars, logspace_flows = False, **kwargs)

            ## The second pass ##
            self.partition_eval(negate_pflows = True)

        else:
            raise NotImplementedError()

        self.samples_consumed += inputs.size(0)

        if update:

            assert self.samples_consumed > 0
            # print(self.samples_consumed)

            self.pc.param_flows[:] /= self.samples_consumed

            for idx, layer in enumerate(self.pc.input_layer_group):
                layer.param_flows[:] /= self.samples_consumed

            self.samples_consumed = 0

            ## Clip gradient ##
            ori_pf = self.pc.param_flows.clone()
            self.pc.param_flows[:] = self.pc.param_flows.clip(min = -5.0, max = 5.0)

            for idx, layer in enumerate(self.pc.input_layer_group):
                layer.param_flows[:] = layer.param_flows.clip(min = -5.0, max = 5.0)

            ## Accum flows ##

            compute_cum_par_flows(self.pc.param_flows, self.pc.parflow_fusing_kwargs)

            for idx, layer in enumerate(self.pc.input_layer_group):

                for i in range(len(layer.tied2source_nids)):
                    pfid_start, num_par_flows, ch_pfids = layer.tied2source_nids[i]
                    num_coalesced_blocks = ch_pfids.size(0)

                    if num_coalesced_blocks <= 1024:
                        BLOCK_N = triton.next_power_of_2(num_coalesced_blocks)
                        BLOCK_M = min(1024 // BLOCK_N, num_par_flows)

                        grid = (triton.cdiv(num_par_flows, BLOCK_M),)

                        layer._pflow_accum_kernel[grid](
                            param_flows_ptr = layer.param_flows,
                            pfid_start = pfid_start,
                            ch_pfids_ptr = ch_pfids,
                            num_coalesced_blocks = num_coalesced_blocks,
                            num_par_flows = num_par_flows,
                            BLOCK_M = BLOCK_M,
                            BLOCK_N = BLOCK_N
                        )
                    else:
                        raise ValueError()

            ## Update parameters ##
            if self.optimizer == "SGD":
                sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs, 
                            lr = lr, keep_zero_params = False, update_log_params = self.update_log_params)

                for idx, layer in enumerate(self.pc.input_layer_group):
                    sgd_input_layer(layer, lr = lr, update_log_params = self.update_log_params)

            elif self.optimizer == "Momentum":
                if self.m is None:
                    self.m = [None for _ in range(len(self.pc.input_layer_group.layers) + 1)]

                gradients = [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

                momentum_update(gradients, self.m, self.t, self.momentum, self.dampening, self.nesterov)

                sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs, 
                               lr = lr, keep_zero_params = False)

                # for idx, layer in enumerate(self.pc.input_layer_group):
                #     sgd_input_layer(layer, lr = lr, update_log_params = self.update_log_params)

            elif self.optimizer == "Adam":
                if self.m is None:
                    self.m = [None for _ in range(len(self.pc.input_layer_group.layers) + 1)]
                    self.v = [None for _ in range(len(self.pc.input_layer_group.layers) + 1)]

                if self.update_log_params:
                    parameters = [self.pc.params] + [layer.params for layer in self.pc.input_layer_group]
                    gradients = [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

                    if self.w_decay == 0.0:
                        adam_update(gradients, self.m, self.v, self.t)
                    else:
                        adam_update_with_decay(parameters, gradients, self.m, self.v, self.t, self.w_decay)
                else:
                    parameters = [self.pc.params] + [layer.params for layer in self.pc.input_layer_group]
                    gradients = [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

                    adam_softplus_update(parameters, gradients, self.m, self.v, self.t)

                # if self.pc.param_flows.isnan().any() or gradients[1].isnan().any():
                #     import pdb; pdb.set_trace()

                sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs, 
                            lr = lr, keep_zero_params = False)

                # if self.pc.params.isnan().any():
                #     import pdb; pdb.set_trace()

                for idx, layer in enumerate(self.pc.input_layer_group):
                    sgd_input_layer(layer, lr = lr, update_log_params = self.update_log_params)

                    # if layer.params.isnan().any():
                    #     import pdb; pdb.set_trace()

            else:
                raise NotImplementedError()

            ## Clip parameters ##
            if not no_param_clipping:
                self.pc.params.clamp_(max = 2000.0)

                for idx, layer in enumerate(self.pc.input_layer_group):
                    layer.params.clamp_(max = 2000.0)

            ## Zero param flows ##
            self.pc.init_param_flows(flows_memory = 0.0)

        return lls

    def apply_update(self, samples_consumed, lr, beta1 = 0.9, beta2 = 0.95):
        device = self.pc.params.device
        with torch.cuda.device(device):
            self.t += 1

            self.pc.param_flows[:] /= samples_consumed

            for idx, layer in enumerate(self.pc.input_layer_group):
                layer.param_flows[:] /= samples_consumed

            ## Clip gradient ##
            ori_pf = self.pc.param_flows.clone()
            self.pc.param_flows[:] = self.pc.param_flows.clip(min = -5.0, max = 5.0)

            for idx, layer in enumerate(self.pc.input_layer_group):
                layer.param_flows[:] = layer.param_flows.clip(min = -5.0, max = 5.0)

            ## Accum flows ##

            compute_cum_par_flows(self.pc.param_flows, self.pc.parflow_fusing_kwargs)

            # import pdb; pdb.set_trace()

            for idx, layer in enumerate(self.pc.input_layer_group):

                for i in range(len(layer.tied2source_nids)):
                    pfid_start, num_par_flows, ch_pfids = layer.tied2source_nids[i]
                    num_coalesced_blocks = ch_pfids.size(0)

                    if num_coalesced_blocks <= 1024:
                        BLOCK_N = triton.next_power_of_2(num_coalesced_blocks)
                        BLOCK_M = min(1024 // BLOCK_N, num_par_flows)

                        grid = (triton.cdiv(num_par_flows, BLOCK_M),)

                        layer._pflow_accum_kernel[grid](
                            param_flows_ptr = layer.param_flows,
                            pfid_start = pfid_start,
                            ch_pfids_ptr = ch_pfids,
                            num_coalesced_blocks = num_coalesced_blocks,
                            num_par_flows = num_par_flows,
                            BLOCK_M = BLOCK_M,
                            BLOCK_N = BLOCK_N
                        )
                    else:
                        raise ValueError()

            ## Update parameters ##
            if self.optimizer == "SGD":
                sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs, 
                            lr = lr, keep_zero_params = False, update_log_params = self.update_log_params)

                for idx, layer in enumerate(self.pc.input_layer_group):
                    sgd_input_layer(layer, lr = lr, update_log_params = self.update_log_params)

            elif self.optimizer == "Momentum":
                if self.m is None:
                    self.m = [None for _ in range(len(self.pc.input_layer_group.layers) + 1)]

                gradients = [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

                momentum_update(gradients, self.m, self.t, self.momentum, self.dampening, self.nesterov)

                sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs, 
                                lr = lr, keep_zero_params = False)

                # for idx, layer in enumerate(self.pc.input_layer_group):
                #     sgd_input_layer(layer, lr = lr, update_log_params = self.update_log_params)

            elif self.optimizer == "Adam":
                if self.m is None:
                    self.m = [None for _ in range(len(self.pc.input_layer_group.layers) + 1)]
                    self.v = [None for _ in range(len(self.pc.input_layer_group.layers) + 1)]

                if self.update_log_params:
                    parameters = [self.pc.params] + [layer.params for layer in self.pc.input_layer_group]
                    gradients = [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

                    # import pdb; pdb.set_trace()

                    if self.w_decay == 0.0:
                        adam_update(gradients, self.m, self.v, self.t, beta1 = beta1, beta2 = beta2)
                    else:
                        adam_update_with_decay(parameters, gradients, self.m, self.v, self.t, self.w_decay)
                else:
                    parameters = [self.pc.params] + [layer.params for layer in self.pc.input_layer_group]
                    gradients = [self.pc.param_flows] + [layer.param_flows for layer in self.pc.input_layer_group]

                    adam_softplus_update(parameters, gradients, self.m, self.v, self.t)

                # if self.pc.param_flows.isnan().any() or gradients[1].isnan().any():
                #     import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()

                sgd_par_update(self.pc.params, self.pc.param_flows, par_update_kwargs = self.pc.par_update_kwargs, 
                            lr = lr, keep_zero_params = False)

                # if self.pc.params.isnan().any():
                #     import pdb; pdb.set_trace()

                for idx, layer in enumerate(self.pc.input_layer_group):
                    sgd_input_layer(layer, lr = lr, update_log_params = self.update_log_params)

                    # if layer.params.isnan().any():
                    #     import pdb; pdb.set_trace()

            else:
                raise NotImplementedError()

            ## Clip parameters ##
            self.pc.params.clamp_(max = 2000.0)

            for idx, layer in enumerate(self.pc.input_layer_group):
                layer.params.clamp_(max = 2000.0)

            ## Zero param flows ##
            self.pc.init_param_flows(flows_memory = 0.0)

    def local_normalize(self):
        
        for idx, layer in enumerate(self.pc.input_layer_group):
            num_cats = layers.nodes[0].dist.num_cats
            inds = layer.s_pids[:,None] + torch.arange(0, num_cats, device = layer.s_pids.device)[None,:]
            layer.params[:] = (layer.params.view(-1, num_cats) / layer.params.view(-1, num_cats).sum(dim = 1, keepdim = True)).reshape(-1)

        normalize_parameters(self.pc.params, self.pc.par_update_kwargs)

    def normalize_by_flows(self, debug = False):
        device = self.pc.params.device
        with torch.cuda.device(device):
            self.pc.init_param_flows(flows_memory = 0.0)

            self.partition_eval(negate_pflows = False, debug = debug)

            # if self.pc.param_flows.isnan().any() or self.pc.input_layer_group[0].param_flows.isnan().any():
            #     import pdb; pdb.set_trace()
            
            self.pc.mini_batch_em(step_size = 1.0, pseudocount = 1e-6)

            self.pc.init_param_flows(flows_memory = 0.0)

    def partition_eval(self, negate_pflows = True, debug = False):

        missing_mask = torch.ones([self.pc.node_mars.size(1), self.pc.num_vars], dtype = bool, device = self.pc.params.device)

        # Forward pass
        self.pc._init_buffer(name = "node_mars", shape = (self.pc.num_nodes, self.pc.node_mars.size(1)), set_value = 0.0)
        self.pc._init_buffer(name = "element_mars", shape = (self.pc.num_elements, self.pc.node_mars.size(1)), set_value = -torch.inf)

        for idx, layer in enumerate(self.pc.input_layer_group):
            nsid, neid = layer._output_ind_range
            num_cats = layer.nodes[0].dist.num_cats

            inds = layer.s_pids[:,None] + torch.arange(0, num_cats, device = self.pc.node_mars.device)[None,:]

            self.pc.node_mars[nsid:neid,:] = layer.params[inds].sum(dim = 1, keepdim = True).log()

        for layer_id, layer_group in enumerate(self.pc.inner_layer_groups):
            if layer_group.is_prod():
                # Prod layer
                layer_group(self.pc.node_mars, self.pc.element_mars)

            elif layer_group.is_sum():
                # Sum layer
                layer_group(self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                            force_use_fp16 = False, force_use_fp32 = False, 
                            propagation_alg = "LL")

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")

        self.pc._init_buffer(name = "node_flows", shape = (self.pc.num_nodes, self.pc.node_mars.size(1)), set_value = -float("inf"))
        self.pc._init_buffer(name = "element_flows", shape = (self.pc.num_elements, self.pc.node_mars.size(1)), set_value = -float("inf"))

        if debug:
            import pdb; pdb.set_trace()

        self.pc.node_flows[self.pc._root_node_range[0]:self.pc._root_node_range[1],:] = 0.0

        for layer_id in range(len(self.pc.inner_layer_groups) - 1, -1, -1):
            layer_group = self.pc.inner_layer_groups[layer_id]

            if layer_group.is_prod():
                # Prod layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, logspace_flows = True)

            elif layer_group.is_sum():
                # Sum layer

                # First recompute the previous product layer
                self.pc.inner_layer_groups[layer_id-1].forward(self.pc.node_mars, self.pc.element_mars, _for_backward = True)

                # Backward sum layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                                        param_flows = self.pc.param_flows, allow_modify_flows = False, propagation_alg = "LL", 
                                        logspace_flows = True, negate_pflows = negate_pflows)

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")

        for idx, layer in enumerate(self.pc.input_layer_group):
            nsid, neid = layer._output_ind_range
            num_cats = layer.nodes[0].dist.num_cats

            if num_cats <= 1024:
                BLOCK_SIZE = triton.next_power_of_2(max(2048 // num_cats, 1))
                BLOCK_C = triton.next_power_of_2(num_cats)
                grid = (triton.cdiv(neid - nsid, BLOCK_SIZE),)

                acc_all_kernel[grid](
                    self.pc.node_flows, layer.s_pfids, layer.s_pids, layer.params, layer.param_flows, 
                    nsid, num_cats = num_cats, batch_size = self.pc.node_mars.size(1), neg = negate_pflows,
                    layer_num_nodes = neid - nsid, BLOCK_SIZE = BLOCK_SIZE, BLOCK_C = BLOCK_C,
                    norm_params = self.input_layer_norm_params
                )
            else:
                BLOCK_SIZE = 8
                BLOCK_C = 256
                nc = triton.cdiv(num_cats, BLOCK_C)

                grid = (triton.cdiv(neid - nsid, BLOCK_SIZE),)

                acc_all_kernel_large[grid](
                    self.pc.node_flows, layer.s_pfids, layer.s_pids, layer.params, layer.param_flows, 
                    nsid, num_cats = num_cats, batch_size = self.pc.node_mars.size(1), neg = negate_pflows,
                    layer_num_nodes = neid - nsid, BLOCK_SIZE = BLOCK_SIZE, BLOCK_C = BLOCK_C, nc = nc, 
                    norm_params = self.input_layer_norm_params
                )

    def partition_eval_fw(self, bn = False, ln = False, record_statistics = False):

        B = self.pc.node_mars.size(1)

        if bn and record_statistics and (not hasattr(self, "bn_mars") or self.bn_mars is None):
            self.bn_mars = torch.zeros([self.pc.num_nodes], device = self.pc.device)

        if ln and record_statistics and (not hasattr(self, "ln_mars") or self.ln_mars is None or self.ln_mars.size(1) != B):
            self.ln_mars = torch.zeros([len(self.pc.root_ns), B], device = self.pc.device)
            self.ns2id = dict()
            for i, ns in enumerate(self.pc.root_ns):
                self.ns2id[ns] = i

        missing_mask = torch.ones([self.pc.node_mars.size(1), self.pc.num_vars], dtype = bool, device = self.pc.params.device)

        # Forward pass
        self.pc._init_buffer(name = "node_mars", shape = (self.pc.num_nodes, self.pc.node_mars.size(1)), set_value = 0.0)
        self.pc._init_buffer(name = "element_mars", shape = (self.pc.num_elements, self.pc.node_mars.size(1)), set_value = -torch.inf)

        for idx, layer in enumerate(self.pc.input_layer_group):
            nsid, neid = layer._output_ind_range
            num_cats = layer.nodes[0].dist.num_cats

            inds = layer.s_pids[:,None] + torch.arange(0, num_cats, device = self.pc.node_mars.device)[None,:]

            self.pc.node_mars[nsid:neid,:] = layer.params[inds].sum(dim = 1, keepdim = True).log()

            # Batch normalization
            if bn:
                lsid, leid = layer._output_ind_range
                if record_statistics:
                    self.bn_mars[lsid:leid] = self.pc.node_mars[lsid:leid,:].mean(dim = 1)
                self.pc.node_mars[lsid:leid,:] -= self.bn_mars[lsid:leid].unsqueeze(1)

            # Layer normalization
            if ln:
                for ns in layer.nodes:
                    nid = self.ns2id[ns]
                    nsid, neid = ns._output_ind_range
                    if record_statistics:
                        self.ln_mars[nid,:] = self.pc.node_mars[nsid:neid,:].mean(dim = 0)
                    self.pc.node_mars[nsid:neid,:] -= self.ln_mars[nid,:].unsqueeze(0)

        for layer_id, layer_group in enumerate(self.pc.inner_layer_groups):
            if layer_group.is_prod():
                # Prod layer
                layer_group(self.pc.node_mars, self.pc.element_mars)

            elif layer_group.is_sum():
                # Sum layer
                layer_group(self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                            force_use_fp16 = False, force_use_fp32 = False, 
                            propagation_alg = "LL")

                # Batch normalization
                if bn:
                    for layer in layer_group:
                        lsid, leid = layer._layer_nid_range
                        if record_statistics:
                            self.bn_mars[lsid:leid] = self.pc.node_mars[lsid:leid,:].mean(dim = 1)
                        self.pc.node_mars[lsid:leid,:] -= self.bn_mars[lsid:leid].unsqueeze(1)

                # Layer normalization
                if ln:
                    for layer in layer_group:
                        for ns in layer.nodes:
                            nid = self.ns2id[ns]
                            nsid, neid = ns._output_ind_range
                            if record_statistics:
                                self.ln_mars[nid,:] = self.pc.node_mars[nsid:neid,:].mean(dim = 0)
                            self.pc.node_mars[nsid:neid,:] -= self.ln_mars[nid,:].unsqueeze(0)

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")

        lls = self.pc.node_mars[self.pc._root_node_range[0]:self.pc._root_node_range[1],0].detach().cpu()

        return lls[0].item()

    def partition_eval_bk(self, ll_weights = None, negate_pflows = True, bn = False, ln = False):
        
        self.pc._init_buffer(name = "node_flows", shape = (self.pc.num_nodes, self.pc.node_mars.size(1)), set_value = -float("inf"))
        self.pc._init_buffer(name = "element_flows", shape = (self.pc.num_elements, self.pc.node_mars.size(1)), set_value = -float("inf"))

        if ll_weights is None:
            self.pc.node_flows[self.pc._root_node_range[0]:self.pc._root_node_range[1],:] = 0.0
        else:
            if ll_weights.dim() == 1:
                ll_weights = ll_weights.unsqueeze(0)
            self.pc.node_flows[self.pc._root_node_range[0]:self.pc._root_node_range[1],:] = ll_weights.log()

        for layer_id in range(len(self.pc.inner_layer_groups) - 1, -1, -1):
            layer_group = self.pc.inner_layer_groups[layer_id]

            if layer_group.is_prod():
                # Prod layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, logspace_flows = True)

            elif layer_group.is_sum():
                # Sum layer

                # Undo batch normalization
                if bn:
                    for layer in layer_group:
                        lsid, leid = layer._layer_nid_range
                        self.pc.node_mars[lsid:leid,:] += self.bn_mars[lsid:leid].unsqueeze(1)

                # Undo layer normalization
                if ln:
                    for layer in layer_group:
                        for ns in layer.nodes:
                            nid = self.ns2id[ns]
                            nsid, neid = ns._output_ind_range
                            self.pc.node_mars[nsid:neid,:] += self.ln_mars[nid,:].unsqueeze(0)

                # First recompute the previous product layer
                self.pc.inner_layer_groups[layer_id-1].forward(self.pc.node_mars, self.pc.element_mars, _for_backward = True)

                # Backward sum layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                                        param_flows = self.pc.param_flows, allow_modify_flows = False, propagation_alg = "LL", 
                                        logspace_flows = True, negate_pflows = negate_pflows)

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")

        for idx, layer in enumerate(self.pc.input_layer_group):

            # Undo batch normalization
            if bn:
                lsid, leid = layer._output_ind_range
                self.pc.node_mars[lsid:leid,:] += self.bn_mars[lsid:leid].unsqueeze(1)

            # Undo layer normalization
            if ln:
                for ns in layer.nodes:
                    nid = self.ns2id[ns]
                    nsid, neid = ns._output_ind_range
                    self.pc.node_mars[nsid:neid,:] += self.ln_mars[nid,:].unsqueeze(0)

            nsid, neid = layer._output_ind_range
            num_cats = layer.nodes[0].dist.num_cats

            if num_cats <= 1024:
                BLOCK_SIZE = triton.next_power_of_2(max(2048 // num_cats, 1))
                BLOCK_C = triton.next_power_of_2(num_cats)
                grid = (triton.cdiv(neid - nsid, BLOCK_SIZE),)

                acc_all_kernel[grid](
                    self.pc.node_flows, layer.s_pfids, layer.s_pids, layer.params, layer.param_flows, 
                    nsid, num_cats = num_cats, batch_size = self.pc.node_mars.size(1), neg = negate_pflows,
                    layer_num_nodes = neid - nsid, BLOCK_SIZE = BLOCK_SIZE, BLOCK_C = BLOCK_C,
                    norm_params = self.input_layer_norm_params
                )
            else:
                BLOCK_SIZE = 8
                BLOCK_C = 256
                nc = triton.cdiv(num_cats, BLOCK_C)

                grid = (triton.cdiv(neid - nsid, BLOCK_SIZE),)

                acc_all_kernel_large[grid](
                    self.pc.node_flows, layer.s_pfids, layer.s_pids, layer.params, layer.param_flows, 
                    nsid, num_cats = num_cats, batch_size = self.pc.node_mars.size(1), neg = negate_pflows,
                    layer_num_nodes = neid - nsid, BLOCK_SIZE = BLOCK_SIZE, BLOCK_C = BLOCK_C, nc = nc,
                    norm_params = self.input_layer_norm_params
                )

    def eval_z(self):
        
        inputs = torch.zeros([self.pc.node_mars.size(1), self.pc.num_vars], dtype = torch.long, device = self.pc.params.device)
        missing_mask = torch.ones([self.pc.node_mars.size(1), self.pc.num_vars], dtype = bool, device = self.pc.params.device)

        # Forward pass
        self.pc._init_buffer(name = "node_mars", shape = (self.pc.num_nodes, inputs.size(0)), set_value = 0.0)
        self.pc._init_buffer(name = "element_mars", shape = (self.pc.num_elements, inputs.size(0)), set_value = -torch.inf)

        for idx, layer in enumerate(self.pc.input_layer_group):
            nsid, neid = layer._output_ind_range
            num_cats = layer.nodes[0].dist.num_cats

            inds = layer.s_pids[:,None] + torch.arange(0, num_cats, device = inputs.device)[None,:]

            self.pc.node_mars[nsid:neid,:] = layer.params[inds].sum(dim = 1, keepdim = True).log()

        for layer_id, layer_group in enumerate(self.pc.inner_layer_groups):
            if layer_group.is_prod():
                # Prod layer
                layer_group(self.pc.node_mars, self.pc.element_mars)

            elif layer_group.is_sum():
                # Sum layer
                layer_group(self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                            force_use_fp16 = False, force_use_fp32 = False, 
                            propagation_alg = "LL")

            else:
                raise ValueError(f"Unknown layer type {type(layer)}.")

        lls = self.pc.node_mars[self.pc._root_node_range[0]:self.pc._root_node_range[1],:]
        lls = lls.permute(1, 0)

        return lls[0,0].detach().cpu().item()

    def batch_norm_fw(self, x):

        if not hasattr(self, "bn_mars") or self.bn_mars is None:
            self.bn_mars = torch.zeros([self.pc.num_nodes], device = self.pc.device)

        B = x.size(0)

        self.pc._init_buffer(name = "node_mars", shape = (self.pc.num_nodes, B), set_value = 0.0)
        self.pc._init_buffer(name = "element_mars", shape = (self.pc.num_elements, B), set_value = -torch.inf)

        x = x.permute(1, 0)

        # Input layers
        for idx, layer in enumerate(self.pc.input_layer_group):
            layer(x, self.pc.node_mars)

            # Batch normalization
            lsid, leid = layer._output_ind_range
            self.bn_mars[lsid:leid] = self.pc.node_mars[lsid:leid,:].mean(dim = 1)
            self.pc.node_mars[lsid:leid,:] -= self.bn_mars[lsid:leid].unsqueeze(1)

        # Inner layers
        for layer_id, layer_group in enumerate(self.pc.inner_layer_groups):
            if layer_group.is_prod():
                # Prod layer
                layer_group(self.pc.node_mars, self.pc.element_mars)

            elif layer_group.is_sum():
                # Sum layer
                layer_group(self.pc.node_mars, self.pc.element_mars, self.pc.params)

                # Batch normalization
                for layer in layer_group:
                    lsid, leid = layer._layer_nid_range
                    self.bn_mars[lsid:leid] = self.pc.node_mars[lsid:leid,:].mean(dim = 1)
                    self.pc.node_mars[lsid:leid,:] -= self.bn_mars[lsid:leid].unsqueeze(1)

        lls = self.pc.node_mars[self.pc._root_node_range[0]:self.pc._root_node_range[1],:]
        lls = lls.permute(1, 0)

        return lls

    def batch_norm_bp(self, x):

        B = x.size(0)

        x = x.permute(1, 0)

        self.pc._init_buffer(name = "node_flows", shape = (self.pc.num_nodes, self.pc.node_mars.size(1)), set_value = -float("inf"))
        self.pc._init_buffer(name = "element_flows", shape = (self.pc.num_elements, self.pc.node_mars.size(1)), set_value = -float("inf"))

        self.pc.node_flows[self.pc._root_node_range[0]:self.pc._root_node_range[1],:] = 0.0

        # Inner layers
        for layer_id in range(len(self.pc.inner_layer_groups) - 1, -1, -1):
            layer_group = self.pc.inner_layer_groups[layer_id]

            if layer_group.is_prod():
                # Prod layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, logspace_flows = True)

            elif layer_group.is_sum():
                # Sum layer

                # Undo batch normalization
                for layer in layer_group:
                    lsid, leid = layer._layer_nid_range
                    self.pc.node_mars[lsid:leid,:] += self.bn_mars[lsid:leid].unsqueeze(1)

                # First recompute the previous product layer
                self.pc.inner_layer_groups[layer_id-1].forward(self.pc.node_mars, self.pc.element_mars, _for_backward = True)

                # Backward sum layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                                     param_flows = self.pc.param_flows, allow_modify_flows = False, propagation_alg = "LL", 
                                     logspace_flows = True, negate_pflows = False)

        # Input layers
        for idx, layer in enumerate(self.pc.input_layer_group):

            # Undo batch normalization
            lsid, leid = layer._output_ind_range
            self.pc.node_mars[lsid:leid,:] += self.bn_mars[lsid:leid].unsqueeze(1)

            layer.backward(x, self.pc.node_flows, self.pc.node_mars, logspace_flows = True)

        return None


    def layer_norm_fw(self, x):

        B = x.size(0)

        if not hasattr(self, "ln_mars") or self.ln_mars is None or self.ln_mars.size(1) != B:
            self.ln_mars = torch.zeros([len(self.pc.root_ns), B], device = self.pc.device)
            self.ns2id = dict()
            for i, ns in enumerate(self.pc.root_ns):
                self.ns2id[ns] = i

        self.pc._init_buffer(name = "node_mars", shape = (self.pc.num_nodes, B), set_value = 0.0)
        self.pc._init_buffer(name = "element_mars", shape = (self.pc.num_elements, B), set_value = -torch.inf)

        x = x.permute(1, 0)

        # Input layers
        for idx, layer in enumerate(self.pc.input_layer_group):
            layer(x, self.pc.node_mars)

            # Layer normalization
            for ns in layer.nodes:
                nid = self.ns2id[ns]
                nsid, neid = ns._output_ind_range
                self.ln_mars[nid,:] = self.pc.node_mars[nsid:neid,:].mean(dim = 0)
                self.pc.node_mars[nsid:neid,:] -= self.ln_mars[nid,:].unsqueeze(0)

        # Inner layers
        for layer_id, layer_group in enumerate(self.pc.inner_layer_groups):
            if layer_group.is_prod():
                # Prod layer
                layer_group(self.pc.node_mars, self.pc.element_mars)

            elif layer_group.is_sum():
                # Sum layer
                layer_group(self.pc.node_mars, self.pc.element_mars, self.pc.params)

                # Layer normalization
                for layer in layer_group:
                    for ns in layer.nodes:
                        nid = self.ns2id[ns]
                        nsid, neid = ns._output_ind_range
                        self.ln_mars[nid,:] = self.pc.node_mars[nsid:neid,:].mean(dim = 0)
                        self.pc.node_mars[nsid:neid,:] -= self.ln_mars[nid,:].unsqueeze(0)

        lls = self.pc.node_mars[self.pc._root_node_range[0]:self.pc._root_node_range[1],:]
        lls = lls.permute(1, 0)

        return lls

    def layer_norm_bp(self, x):

        B = x.size(0)

        x = x.permute(1, 0)

        self.pc._init_buffer(name = "node_flows", shape = (self.pc.num_nodes, self.pc.node_mars.size(1)), set_value = -float("inf"))
        self.pc._init_buffer(name = "element_flows", shape = (self.pc.num_elements, self.pc.node_mars.size(1)), set_value = -float("inf"))

        self.pc.node_flows[self.pc._root_node_range[0]:self.pc._root_node_range[1],:] = 0.0

        # Inner layers
        for layer_id in range(len(self.pc.inner_layer_groups) - 1, -1, -1):
            layer_group = self.pc.inner_layer_groups[layer_id]

            if layer_group.is_prod():
                # Prod layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, logspace_flows = True)

            elif layer_group.is_sum():
                # Sum layer

                # Undo layer normalization
                for layer in layer_group:
                    for ns in layer.nodes:
                        nid = self.ns2id[ns]
                        nsid, neid = ns._output_ind_range
                        self.pc.node_mars[nsid:neid,:] += self.ln_mars[nid,:].unsqueeze(0)

                # First recompute the previous product layer
                self.pc.inner_layer_groups[layer_id-1].forward(self.pc.node_mars, self.pc.element_mars, _for_backward = True)

                # Backward sum layer
                layer_group.backward(self.pc.node_flows, self.pc.element_flows, self.pc.node_mars, self.pc.element_mars, self.pc.params, 
                                     param_flows = self.pc.param_flows, allow_modify_flows = False, propagation_alg = "LL", 
                                     logspace_flows = True, negate_pflows = False)

        # Input layers
        for idx, layer in enumerate(self.pc.input_layer_group):

            # Undo layer normalization
            for ns in layer.nodes:
                nid = self.ns2id[ns]
                nsid, neid = ns._output_ind_range
                self.pc.node_mars[nsid:neid,:] += self.ln_mars[nid,:].unsqueeze(0)

            layer.backward(x, self.pc.node_flows, self.pc.node_mars, logspace_flows = True)

        return None
