import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunedLlamaMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x,                  # (B, S, H)
                gate_w, gate_b,     # gate_proj: w.shape=(I,H), b.shape=(I,)
                up_w,   up_b,       # up_proj:   w.shape=(I,H), b.shape=(I,)
                down_w, down_b,     # down_proj: w.shape=(H,I), b.shape=(H,)
                prune_ratio=0.5):
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
        _, top_indices = torch.topk(col_norm_sum, keep_num, largest=True)
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
            None            # dL/d(prune_ratio)
        )