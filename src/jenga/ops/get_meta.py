import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_lower_triangle(
    Q, K,
    # 维度信息
    B, H, N_CTX,
    # 输出
    Out,
    # strides
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_ob, stride_om, stride_on,
    # 常量
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    HEAD_DIM: tl.constexpr,
    INV_SQRT_HEAD_DIM: tl.constexpr,
):
    """
    计算近似于：
      attn_weights = (Q @ K^T) / sqrt(HEAD_DIM)
      只保留下三角(row >= col)，否则置0
      再做 ReLU
      所有 head 的结果累加到 acc
      对 acc 做 64x64 的块内 reduce-max
      存到 Out[b, row_block_idx, col_block_idx]
    """
    # 计算当前块在行、列、batch 方向的标识
    row_block_idx = tl.program_id(0)  # 对应行方向(序列的维度)的分块
    col_block_idx = tl.program_id(1)  # 对应列方向(序列的维度)的分块
    b_idx         = tl.program_id(2)  # 对应 batch 的分块

    # 计算本块在行/列方向的具体范围
    row_start = row_block_idx * BLOCK_M
    col_start = col_block_idx * BLOCK_N

    # 在块内的偏移
    offs_m = tl.arange(0, BLOCK_M)  # 0..63
    offs_n = tl.arange(0, BLOCK_N)  # 0..63
    offs_d = tl.arange(0, HEAD_DIM)

    # 维护一个 shape=[BLOCK_M, BLOCK_N] 的 accumulator，用 float32 避免中间溢出
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 遍历所有 head，做 QK^T
    # 注：为了演示，假设 HEAD_DIM 不算太大，一次性 dot；若 HEAD_DIM 很大可再分块
    for h_idx in range(0, H):
        # 1) 取出 Q_block: [BLOCK_M, HEAD_DIM]
        q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh \
                   + (row_start + offs_m[:, None]) * stride_qm \
                   + offs_d[None, :] * stride_qk
        q_block = tl.load(q_ptrs)  # [BLOCK_M, HEAD_DIM], dtype与Q相同(float16/bf16)

        # 2) 取出 K_block: [HEAD_DIM, BLOCK_N]
        k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh \
                   + offs_d[:, None] * stride_km \
                   + (col_start + offs_n[None, :]) * stride_kk
        k_block = tl.load(k_ptrs)  # [HEAD_DIM, BLOCK_N]

        # 3) dot => [BLOCK_M, BLOCK_N]
        partial = tl.dot(q_block, k_block)  # float32

        # 4) / sqrt(HEAD_DIM) - 注意这里改为乘以 INV_SQRT_HEAD_DIM
        partial = partial * INV_SQRT_HEAD_DIM

        # 5) 只保留下三角: row >= col
        cond = (row_start + offs_m[:, None]) >= (col_start + offs_n[None, :])
        partial = tl.where(cond, partial, 0.0)

        # 6) ReLU
        partial = tl.maximum(partial, 0.0)

        # 7) 累加到 acc
        acc += partial

    # acc 形状是 [BLOCK_M, BLOCK_N]
    # 做块内的 reduce-max (max-pool kernel=64, stride=64)
    row_max = tl.max(acc, axis=1)      # [BLOCK_M]
    block_max = tl.max(row_max, axis=0)  # 标量

    # 写入输出张量
    out_ptrs = Out + b_idx * stride_ob \
                     + row_block_idx * stride_om \
                     + col_block_idx * stride_on
    tl.store(out_ptrs, block_max.to(Out.type.element_ty))


def block_lower_triangle_attn_pool(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    计算相当于:
        attn_weights = (q @ k^T) / sqrt(D)
        只保留下三角 (row >= col)，再对每个head做ReLU，最后在head维度上累加
        得到 [B, N, N]
        然后对 [N,N] 做 kernel=64, stride=64 的 2D max-pool (分块法)
        输出 shape = [B, N//64, N//64]

      - q.shape = [B, H, N, D]
      - k.shape = [B, H, D, N]  (在外部已经转好了)
      - q, k 可以是 float16/bfloat16
      - 本函数不再除以 64
    """
    assert q.is_cuda and k.is_cuda, "Triton kernel only works on GPU."
    B, H, N, Dq = q.shape
    Bk, Hk, Dk, Nk = k.shape
    assert B == Bk and H == Hk and N == Nk and Dq == Dk, "q, k shape mismatch."
    BLOCK = 64
    assert N % BLOCK == 0, "For simplicity, assume N is multiple of 64."

    # 创建输出张量: [B, N//BLOCK, N//BLOCK], 用 float32
    out = torch.empty((B, N // BLOCK, N // BLOCK), dtype=torch.float32, device=q.device)

    # 计算 strides
    stride_qb, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kh, stride_km, stride_kk = k.stride()
    stride_ob, stride_om, stride_on = out.stride()

    # 先在 Python 里把 1.0/sqrt(Dq) 算好，传给 Kernel
    inv_sqrt_d = float(1.0 / math.sqrt(Dq))

    # 定义 grid: (gridM, gridN, gridB)
    grid = (N // BLOCK, N // BLOCK, B)

    # 调用 Triton kernel
    _fwd_kernel_lower_triangle[grid](
        # 数据
        q, k,
        # 尺寸
        B, H, N,
        # 输出
        out,
        # strides
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_km, stride_kk,
        stride_ob, stride_om, stride_on,
        # 常量
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        HEAD_DIM=Dq,              # 传递给 Kernel 作编译时常量
        INV_SQRT_HEAD_DIM=inv_sqrt_d,  # 同上
        num_stages=2,
        num_warps=4,
    )

    return out


class PrunedLlamaMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x,                  # (B, S, H)
                gate_w, gate_b,     # gate_proj: w.shape=(I,H), b.shape=(I,)
                up_w,   up_b,       # up_proj:   w.shape=(I,H), b.shape=(I,)
                down_w, down_b,     # down_proj: w.shape=(H,I), b.shape=(H,)
                prune_ratio,
                layer_idx,
                show_mlp=False):
        """
        x:          (B, S, H),  可能需要对 x 求梯度 (LoRA之外的部分)
        gate_w,b:   gate_proj 的冻结权重/偏置
        up_w,b:     up_proj   的冻结权重/偏置
        down_w,b:   down_proj 的冻结权重/偏置
        prune_ratio: 裁剪比例，0.5 => 保留一半列
        """

        # ---------- 0. 基本信息 ----------
        B, S, H = x.shape                                # batch_size, seq_len, hidden_size
        I = gate_w.shape[0]                              # intermediate_size

        # ========== (1) gate = gate_proj(x) ==========
        #    gate = x @ gate_w^T + gate_b
        #    形状: (B, S, I)
        gate = torch.matmul(x, gate_w.t())

        # ========== (2) up = up_proj(x) ==========
        #    up = x @ up_w^T + up_b
        #    形状: (B, S, I)
        up = torch.matmul(x, up_w.t()) 

        # ========== (3) 对 gate, up 同时做 "公共 Top-K" 裁剪 ==========
        #    先把 (B, S, I) reshape 成 (B*S, I)
        #    col_norm_gate[i] =  || gate[:, :, i] ||2
        #    col_norm_up[i]   =  || up[:, :, i]   ||2
        a = F.silu(gate)
        gate_2d = a.reshape(-1, I)   # (B*S, I)
        up_2d   = up.reshape(-1, I)     # (B*S, I)
        col_norm_gate = gate_2d.norm(p=2, dim=0)  # (I,)
        col_norm_up   = up_2d.norm(p=2, dim=0)    # (I,)

        #    合并: col_norm_sum = col_norm_gate + col_norm_up
        col_norm_sum = col_norm_gate + col_norm_up
        keep_num = int(I * (1 - prune_ratio))  # 比如 prune_ratio=0.5 => keep_num= I/2

        #    选出最大的 keep_num 个列 index
        top_values, top_indices = torch.topk(col_norm_sum, keep_num, largest=True)
        min_value = top_values[-1]  # 选出后的最小值
        max_value = top_values[0]   # 选出后的最大值
        if show_mlp:
            print(f"layer: {layer_idx} , threshold: {min_value/max_value:.4f}, memory: {1-prune_ratio}")
        #    裁剪:
        #      gate_small: (B, S, keep_num)
        #      up_small  : (B, S, keep_num)
        gate_small = torch.index_select(gate, dim=2, index=top_indices)
        up_small   = torch.index_select(up,   dim=2, index=top_indices)
        a_small    = torch.index_select(a,    dim=2, index=top_indices)

        # ========== (4) a_small = SiLU(gate_small) ==========
        #    a_small: (B, S, keep_num)
        # a_small = F.silu(gate_small)

        # ========== (5) c_small = a_small * up_small ==========
        #    逐元素乘 => (B, S, keep_num)
        # c_small = gate_small * up_small
        c = a * up
        # ========== (6) 还原回 c: (B, S, I) ==========
        # c = torch.zeros_like(gate)  # same shape (B, S, I)
        # c = c.index_copy(dim=2, index=top_indices, source=c_small)

        # ========== (7) out = down_proj(c) ==========
        #    out = c @ down_w^T + down_b
        #    shape: (B, S, H)
        out = torch.matmul(c, down_w.t()) 
        # ---------- 保存到 ctx，以备 backward ----------
        # 虽然权重是冻结的，但为了计算 dL/dx，需要它们做矩阵乘法
        # PyTorch 对于叶子节点 + requires_grad=False 的权重，一般只会存 "引用" 而不是复制
        ctx.save_for_backward(               # (B, S, H)
            gate_small, up_small, a_small,       # (B, S, keep_num)
            top_indices,     # (keep_num,)
            gate_w, 
            up_w,   
            down_w
        )
        ctx.prune_ratio = prune_ratio
        ctx.B = x.shape[0]
        ctx.S = x.shape[1]
        ctx.H = x.shape[2]
        ctx.I = I
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: (B, S, H)   -- 对 out 的梯度
        返回与 forward() 同样顺序:
          dL/d(x), dL/d(gate_w), dL/d(gate_b), dL/d(up_w), dL/d(up_b), dL/d(down_w), dL/d(down_b), dL/d(prune_ratio)
        但是权重是冻结的 => 我们返回 None
        并注释清楚每一步维度以及在算什么。
        """
        (gate_small, up_small, a_small,
         top_indices,
         gate_w, 
         up_w,   
         down_w, ) = ctx.saved_tensors

        prune_ratio = ctx.prune_ratio
        B, S, H, I= ctx.B, ctx.S, ctx.H, ctx.I
        keep_num = int(I * (1 - prune_ratio))

        #--------------------------------------------------
        # 反向传播各步 (带维度注释)
        #--------------------------------------------------

        # ========== (7) out = c @ down_w^T + down_b ==========
        #   out.shape = (B, S, H)
        #   => grad_c = grad_out @ down_w   [ (B,S,H) x (H,I) = (B,S,I) ]
        grad_c = torch.matmul(grad_out, down_w)   # shape (B, S, I)

        # ========== (6) c = index_copy => c_small = a_small * up_small ==========
        #   => grad_c_small = grad_c[:, :, top_indices]
        grad_c_small = torch.index_select(grad_c, dim=2, index=top_indices)   # shape (B,S,keep_num)

        #   c_small = a_small * up_small
        #      => grad_a_small = grad_c_small * up_small
        #      => grad_up_small = grad_c_small * a_small
        grad_a_small = grad_c_small * up_small      # (B,S,keep_num)
        grad_up_small= grad_c_small * a_small       # (B,S,keep_num)

        # ========== (5) a_small = SiLU(gate_small) ==========
        #   => grad_gate_small = grad_a_small * silu'(gate_small)
        #   其中 gate_small = gate[:, :, top_indices]
        #   silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))) ... 也可用内置
        with torch.enable_grad():
            gate_small_ = gate_small.detach().clone().requires_grad_(True)
            a_small_    = F.silu(gate_small_)
            # 对 a_small_.sum() 做求导 => grad_gate_small_
            grad_gate_small_ = torch.autograd.grad(a_small_.sum(), gate_small_, retain_graph=False)[0]
        grad_gate_small = grad_a_small * grad_gate_small_

        # ========== (3)(4) gate_small, up_small => index_select 的逆过程 ==========
        #   grad_gate: (B,S,I) 全0 + index_copy 回 grad_gate_small
        #   grad_up:   (B,S,I) 同理
        grad_gate = torch.zeros([B,S,I],dtype=grad_gate_small.dtype,device=grad_gate_small.device)  # (B,S,I)
        grad_up   = torch.zeros([B,S,I],dtype=grad_gate_small.dtype,device=grad_gate_small.device)
        grad_gate.index_copy_(2, top_indices, grad_gate_small)
        grad_up.index_copy_(2, top_indices, grad_up_small)

        # ========== (1) gate = x @ gate_w^T + gate_b ==========
        #   => grad_x_from_gate, grad_gate_w, grad_gate_b


        grad_x_from_gate = torch.matmul(grad_gate, gate_w)  # (B,S,H)

        # ========== (2) up = x @ up_w^T + up_b ==========

        grad_x_from_up = torch.matmul(grad_up, up_w)  # (B,S,H)

        # ========== 合并 dL/dx = grad_x_from_gate + grad_x_from_up ==========
        grad_x = grad_x_from_gate + grad_x_from_up

        #--------------------------------------------------
        # 由于权重是冻结的 => 全部返回 None
        # 如果你想要 0 而不是 None，也行；但是一般直接 None 即可。
        # 只对 x 保留真正的梯度(如果 x.requires_grad=True)。
        #--------------------------------------------------
        return (
            grad_x,         # dL/d(x)
            None, None,     # dL/d(gate_w), dL/d(gate_b)
            None, None,     # dL/d(up_w),   dL/d(up_b)
            None, None,     # dL/d(down_w), dL/d(down_b)
            None, None        # dL/d(prune_ratio)
        )