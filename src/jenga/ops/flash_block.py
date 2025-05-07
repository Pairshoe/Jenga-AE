import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
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
    HEAD_DIM: tl.constexpr, # 128
):
    """
    对给定的 (b, row_block, col_block) 计算:
        1) 对所有 head 做局部的 QK^T
        2) ReLU
        3) 将各 head 的结果相加
        4) 计算本块的最大值
        5) 将最大值写入到 Out[b, row_block, col_block]
       然后调用方再做 /64 即可。

    参数说明：
      Q: float16/bfloat16, shape = [B, H, N_CTX, HEAD_DIM]
      K: float16/bfloat16, shape = [B, H, N_CTX, HEAD_DIM]  (与 Q 同形状，但要做 K^T 时可能需要你在外部处理)
      B, H, N_CTX, HEAD_DIM: batch大小, head数, 序列长度, head_dim
      Out: float32, shape=[B, N_CTX//64, N_CTX//64]
      stride_qb, stride_qh, stride_qm, stride_qk: Q的stride
      stride_kb, stride_kh, stride_km, stride_kk: K的stride
      stride_ob, stride_om, stride_on: Out的stride
      BLOCK_M, BLOCK_N: 本kernel使用的分块大小(一般都是 64)
    """

    # 计算当前块在行、列、batch 方向的标识
    row_block_idx = tl.program_id(0)  # 对应行方向(序列的维度)的分块
    col_block_idx = tl.program_id(1)  # 对应列方向(序列的维度)的分块
    b_idx         = tl.program_id(2)  # 对应 batch 的分块（这里简化为1D，如果B较大可再做更多拆分）
    
    # 计算本块在行/列方向的具体范围
    row_start = row_block_idx * BLOCK_M
    col_start = col_block_idx * BLOCK_N

    # 在块内的偏移
    offs_m = tl.arange(0, BLOCK_M)  # 0..63
    offs_n = tl.arange(0, BLOCK_N)  # 0..63
    offs_dim = tl.arange(0, HEAD_DIM)

    # 维护一个 shape=[BLOCK_M, BLOCK_N] 的accumulator，用float32来避免中间溢出
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 遍历所有 head，并对 QK^T 做分块
    # 这里示例写成简单的 for 循环：每个 block 并不拆分 HEAD_DIM，而是直接一次性做 matmul
    # 如果 HEAD_DIM=64/128，这样做问题不大。若 HEAD_DIM 很大，需要再行分块。
    for h_idx in range(0, H):
        # 取 Q_block, 形状: [BLOCK_M, HEAD_DIM]
        # 注意 Q 的存储布局是 [b, h, m, d]，我们要对 row_start + offs_m 这些行取数据
        # 计算地址：Q + b_idx*stride_qb + h_idx*stride_qh + (row_start+offs_m)*stride_qm + 0*stride_qk
        q_ptrs = Q + b_idx*stride_qb + h_idx*stride_qh \
                   + (row_start + offs_m[:, None]) * stride_qm \
                   + offs_dim[None, :]  * stride_qk
        # 载入 Q_block
        # shape = [BLOCK_M, HEAD_DIM]，但在 Triton 中可以把 (BLOCK_M, HEAD_DIM) 看成 (BLOCK_M, 1) 的 2D + for-loop
        # 这里为简便，直接一次性加载
        q_block = tl.load(q_ptrs)  # [BLOCK_M, HEAD_DIM]

        # 取 K_block, 形状: [HEAD_DIM, BLOCK_N]
        # K 的存储是 [b, h, m, d]，但要注意这是 K，而我们这里要做 K^T 的乘法 => Q * K^T 
        # 最简单方法：K 若是 [b,h,N_CTX,HEAD_DIM]，那 K^T 就是 [b,h,HEAD_DIM,N_CTX] => 读的时候把 (m, d) 对调成 (d, m) 即可
        # 计算地址：K + b_idx*stride_kb + h_idx*stride_kh + 0*stride_kk + (col_start + offs_n)*stride_km
        #         这里 (d, m) => (offs_k, offs_n) 但是因为 HEAD_DIM 方向我们一次性都加载 => offs_k 用个 range?
        # 为了简单，这里示例不再演示“严格意义上的 K^T”。假设 K 的 shape 就是 [B,H,HEAD_DIM,N_CTX]，那 strides 不同。
        # 如果还是 [B,H,N_CTX,HEAD_DIM]，要么在外面先转置 K，要么在这里算地址时做翻转。灵活处理即可。
        # 下面假设 K 已经转置好：K.shape = [B,H,HEAD_DIM,N_CTX] => stride_km = stride_kk, stride_kk= stride_km ?
        # 为了让示例代码更直观，这里直接把 K 当成 [B,H,HEAD_DIM,N_CTX] 用，读取 (d, col) = (0..HEAD_DIM, col_start+offs_n)
        
        k_ptrs = K + b_idx*stride_kb + h_idx*stride_kh \
                   + offs_dim[:, None] * stride_km \
                   + (col_start + offs_n[None, :]) * stride_kk
        # 载入 k_block => shape=[HEAD_DIM, BLOCK_N]
        k_block = tl.load(k_ptrs)

        # 做 matmul => [BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N] => [BLOCK_M, BLOCK_N]
        partial = tl.dot(q_block, k_block)  # float32

        # 每个 head 做完 matmul 后，需要对 partial 做 ReLU，再累加到 acc
        partial = tl.maximum(partial, 0.0)
        acc += partial

    # 此时 acc = \sum_{h=0}^{H-1} ReLU(Q@K^T)   形状是 [BLOCK_M, BLOCK_N]
    # 最终需要对这个 64x64 块求最大值，代表 max-pool(kernel=64, stride=64) 这一块
    # 下面做块内 reduce-max => shape [1]
    # Triton 里可以手动写 for-loop reduce，也可以用内置 reduce 运算
    row_max = tl.max(acc, axis=1)
    block_max = tl.max(row_max, axis=0) # 先对 axis=1 reduce => 每行最大值，再对axis=0 reduce => 全块最大值

    # 写入输出张量 Out[b, row_block_idx, col_block_idx] = block_max
    # Out 的 shape = [B, N_CTX//64, N_CTX//64]
    # 对应地址：Out + b_idx*stride_ob + row_block_idx*stride_om + col_block_idx*stride_on
    out_ptrs = Out + b_idx*stride_ob + row_block_idx*stride_om + col_block_idx*stride_on
    tl.store(out_ptrs, block_max.to(Out.type.element_ty))


def block_attn_pool(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    利用 Triton 分块计算近似于:
        attn_weight = ReLU(Q @ K^T)  (逐个 head 做 ReLU，再累加)
        -> shape [B, N, N], 最后对 [N, N] 做 kernel=64, stride=64 的 max-pool，再 /64
    返回张量形状 [B, N//64, N//64]。

    要求:
      - q.shape = [B, H, N, HEAD_DIM]
      - k.shape = [B, H, HEAD_DIM, N]  # 假设外部已经对 k 做了转置, 以方便 Q@K^T
      - q, k 都是半精度(float16 / bf16)或更低精度
    """
    assert q.is_cuda and k.is_cuda, "Triton kernel only works on GPU tensors."
    B, H, N, Dq = q.shape
    Bk, Hk, Dk, Nk = k.shape
    assert B == Bk and H == Hk and N == Nk and Dq == Dk, "q, k shape mismatch."
    BLOCK = 64
    assert N % BLOCK == 0, "For simplicity, assume N is multiple of 64."

    # 创建输出张量: [B, N//64, N//64], float32 容纳 max-pool 结果
    out = torch.empty((B, N // BLOCK, N // BLOCK), dtype=torch.bfloat16, device=q.device)

    # 计算 strides（以元素为单位，而不是字节）
    # Q: shape [B, H, N, D]
    #   q.stride() 可能返回形如 (H*N*D, N*D, D, 1)，具体看 PyTorch 默认布局
    stride_qb = q.stride(0)
    stride_qh = q.stride(1)
    stride_qm = q.stride(2)
    stride_qk = q.stride(3)
    # K: shape [B, H, D, N]
    stride_kb = k.stride(0)
    stride_kh = k.stride(1)
    stride_km = k.stride(2)
    stride_kk = k.stride(3)
    # Out: shape [B, N//64, N//64]
    stride_ob = out.stride(0)
    stride_om = out.stride(1)
    stride_on = out.stride(2)

    # 定义 grid: (gridM, gridN, gridB)
    #   - gridM = N//64, 逐块遍历行方向
    #   - gridN = N//64, 逐块遍历列方向
    #   - gridB = B,     逐个 batch 处理
    grid = (N // BLOCK, N // BLOCK, B)

    # 调用 Triton kernel
    _fwd_kernel[grid](
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
        HEAD_DIM=128,  # 这里假设 HEAD_DIM=128
        num_stages=2,  # 这里可根据需求调优
        num_warps=4,   # 这里可根据需求调优
    )

    # 根据需求，最后还要做一次 / 64
    out /= 64.0
    return out


def demo():
    """
    简单演示：随机初始化 q, k，手动计算参考值，对比 Triton 分块结果。
    """
    B, H, N, D = 1, 32, 1024, 128  # 尺寸小一点方便演示
    device = "cuda"

    # 准备 Q, K, 并把 K 转成 [B,H,D,N] 形式，以便 Q@K^T
    #   => 这样 Q.shape=[B,H,N,D], K.shape=[B,H,D,N]
    # 这样 Q@K^T 就是 [N,D] x [D,N] => [N,N] (对同一head)
    # 同时最外面还有 B,H 两个维度。
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)
    # k原本 [B,H,N,D]，先转置(-2,-1) => [B,H,D,N]
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)

    # Triton 分块结果
    out_triton = block_attn_pool(q, k.transpose(2,3).contiguous())  # [B, N//64, N//64]

    # 下面做一个参考计算(非分块，直接 matmul)，并比较结果
    # 参考: attn_weight = sum_{h} ReLU( Q[h] @ K[h].T )   => shape = [B,N,N]
    #       之后对 [N,N] 做 kernel=64,stride=64 的 max-pool => [N//64, N//64]
    #       再除以 64
    # with torch.no_grad():
    #     # [B, H, N, N]
    #     attn = torch.zeros((B, N, N), dtype=torch.float32, device=device)
    #     for b_idx in range(B):
    #         for h_idx in range(H):
    #             q_ = q[b_idx, h_idx]            # [N, D]
    #             k_ = k[b_idx, h_idx]            # [D, N]
    #             partial = q_ @ k_              # => [N, N], float32
    #             partial = torch.maximum(partial, torch.tensor(0.0, device=device))  # ReLU
    #             attn[b_idx] += partial

    #     # attn: [B, N, N]
    #     # max_pool2d(kernel=64, stride=64):  => [B, N//64, N//64]
    #     # PyTorch 要求输入是 [B, C, H, W] 形式，这里可以把 attn视为 (B,1,N,N)，再 pool
    #     attn_4d = attn.unsqueeze(1)  # [B, 1, N, N]
    #     pool_ref = torch.nn.functional.max_pool2d(attn_4d, kernel_size=64, stride=64)  # [B,1,N//64,N//64]
    #     pool_ref = pool_ref.squeeze(1)  # [B, N//64, N//64]
    #     pool_ref /= 64.0
    
    attn_weight = torch.matmul(q, k.transpose(-2,-1))    # [B, H, N, N]
    for i in range(attn_weight.size(1)):                 # 对每个 head 做 ReLU
        attn_weight[:, i][attn_weight[:, i] < 0] = 0
    attn_weight = attn_weight.sum(dim=1)                 # 在 head 维度上求和 -> [B, N, N]
    attn_maxpool = torch.nn.functional.max_pool2d(attn_weight, kernel_size=64, stride=64)  # [B, N//64, N//64]
    attn_maxpool /= 64

    # 对比误差
    diff = (out_triton - attn_maxpool).abs().max().item()
    print(f"max abs diff = {diff:.6f}")
    if diff < 1e-2:
        print("Test PASS!")
    else:
        print("Test FAIL!")
    print(out_triton)
    print(attn_maxpool)


if __name__ == "__main__":
    demo()