import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个两层的 MLP
class AttnPredictor(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(AttnPredictor, self).__init__()
        self.fc1 = nn.Linear(dim*64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim*64)
        self.fc3 = nn.Linear(dim*64, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class AttnPredictor1(nn.Module):
    def __init__(self, dim, hidden_dim, n_head=32):
        super(AttnPredictor1, self).__init__()
        self.qlinear1 = nn.Linear(dim*64, hidden_dim)
        self.qlinear2 = nn.Linear(hidden_dim, dim*64)
        self.qlinear3 = nn.Linear(dim*64, dim)

        self.klinear1 = nn.Linear(dim*64, hidden_dim)
        self.klinear2 = nn.Linear(hidden_dim, dim*64)
        self.klinear3 = nn.Linear(dim*64, dim)
        
        self.n_head = n_head
    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        hidden_states = hidden_states.view(bsz, q_len, self.n_head, -1).transpose(1, 2)
        hidden_states = hidden_states.reshape(bsz, self.n_head, q_len//64, -1)
        
        qx = F.relu(self.qlinear1(hidden_states))
        qx = F.relu(self.qlinear2(qx))
        qx = self.qlinear3(qx)
        
        kx = F.relu(self.klinear1(hidden_states))
        kx = F.relu(self.klinear2(kx))
        kx = self.klinear3(kx)
        
        attn = torch.matmul(qx, kx.transpose(-1, -2))
        attn = attn.sum(dim=1)
        
        # attn *= 64
        return attn
    
    
class PrunableAttnPredictor(nn.Module):
    """
    对 (qlinear1, qlinear2, klinear1, klinear2) 做结构化剪枝，
    并在剪完后同步更新 dead_count_xx 缓冲的形状，避免后续前向出现维度不匹配。
    """
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        q1_outdim=None, 
        q2_outdim=None, 
        k1_outdim=None, 
        k2_outdim=None, 
        n_head=32
    ):
        super(PrunableAttnPredictor, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        
        # 若传入 q1_outdim 等为 None，表示尚未剪枝，初始化为默认形状
        q1_out = q1_outdim if q1_outdim is not None else hidden_dim
        q2_out = q2_outdim if q2_outdim is not None else (dim * 64)
        k1_out = k1_outdim if k1_outdim is not None else hidden_dim
        k2_out = k2_outdim if k2_outdim is not None else (dim * 64)

        # ============== Q 路径 ==============
        self.qlinear1 = nn.Linear(dim*64, q1_out,bias=False)
        self.qlinear2 = nn.Linear(q1_out, q2_out,bias=False)
        self.qlinear3 = nn.Linear(q2_out, dim,bias=False)

        # ============== K 路径 ==============
        self.klinear1 = nn.Linear(dim*64, k1_out,bias=False)
        self.klinear2 = nn.Linear(k1_out, k2_out,bias=False)
        self.klinear3 = nn.Linear(k2_out, dim,bias=False)

        # ====== 注册缓冲，用于统计每个线性层的死神经元数量 ======
        # shape 都是 [out_features]，因为我们统计的是“out_features 维度上被激活置零”的次数
        self.register_buffer("dead_count_q1", torch.zeros(q1_out))
        self.register_buffer("dead_count_q2", torch.zeros(q2_out))
        self.register_buffer("dead_count_k1", torch.zeros(k1_out))
        self.register_buffer("dead_count_k2", torch.zeros(k2_out))

        # 记录 forward 调用的次数
        self.forward_steps = 0
        # 记录累计激活数(可选，用于计算死亡率)
        self.total_count = 0

    def forward(self, hidden_states):
        """
        hidden_states 形状: [batch_size, seq_len, hidden_dim], 
        其中 hidden_dim = self.dim * self.n_head (仅示例).
        """
        bsz, q_len, _ = hidden_states.size()

        # 你的 reshape 逻辑 (示例)
        hidden_states = hidden_states.view(bsz, q_len, self.n_head, -1).transpose(1, 2)
        hidden_states = hidden_states.reshape(bsz, self.n_head, q_len // 64, -1)

        # ============ Q 路径 ============
        qx1 = self.qlinear1(hidden_states)
        with torch.no_grad():
            # 统计本次 batch 在 qlinear1 输出上的激活总量(可选)
            self.total_count += qx1.size(0)*qx1.size(1)*qx1.size(2)
            # 统计 qlinear1 输出 <= 0 的次数
            dead_mask_q1 = (qx1 <= 0).sum(dim=(0, 1, 2))
            self.dead_count_q1 += dead_mask_q1
        qx1 = F.relu(qx1)

        qx2 = self.qlinear2(qx1)
        with torch.no_grad():
            dead_mask_q2 = (qx2 <= 0).sum(dim=(0, 1, 2))
            self.dead_count_q2 += dead_mask_q2
        qx2 = F.relu(qx2)

        qx = self.qlinear3(qx2)

        # ============ K 路径 ============
        kx1 = self.klinear1(hidden_states)
        with torch.no_grad():
            dead_mask_k1 = (kx1 <= 0).sum(dim=(0, 1, 2))
            self.dead_count_k1 += dead_mask_k1
        kx1 = F.relu(kx1)

        kx2 = self.klinear2(kx1)
        with torch.no_grad():
            dead_mask_k2 = (kx2 <= 0).sum(dim=(0, 1, 2))
            self.dead_count_k2 += dead_mask_k2
        kx2 = F.relu(kx2)

        kx = self.klinear3(kx2)

        # 计算 attn
        attn = torch.matmul(qx, kx.transpose(-1, -2))
        attn = attn.sum(dim=1)

        self.forward_steps += 1
        return attn

    def _prune_one_linear(self, linear_layer: nn.Linear, dead_count: torch.Tensor, zero_ratio_threshold: float):
        """
        根据 dead_count 得到死亡率 mask，修改 out_features。
        返回 (new_linear, mask)。
        """
        # 计算死亡率
        # 若 self.total_count = 0, 说明尚未统计到数据; 做个保护
        if self.total_count == 0:
            return linear_layer, None

        death_ratio = dead_count / float(self.total_count)
        # 生成保留/剪除 mask
        mask = (death_ratio < zero_ratio_threshold)
        old_out_dim = linear_layer.out_features
        num_to_keep = mask.sum().item()

        # 若全部死/全部活，就不剪
        if num_to_keep == 0 or num_to_keep == old_out_dim:
            print(f"[Warning] skip pruning: keep={num_to_keep}/{old_out_dim}")
            return linear_layer, None

        # 创建新的 linear
        new_linear = nn.Linear(
            in_features=linear_layer.in_features,
            out_features=num_to_keep,
            bias=linear_layer.bias is not None
        ).to(linear_layer.weight.device).to(linear_layer.weight.dtype)

        with torch.no_grad():
            new_linear.weight.copy_(linear_layer.weight[mask])
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias[mask])

        print(f"[Pruning] {linear_layer.__class__.__name__} out_features {old_out_dim} -> {num_to_keep}")
        return new_linear, mask

    def _prune_next_layer_in(self, linear_layer: nn.Linear, mask: torch.Tensor):
        """
        根据上游 out_features 的 mask，剪裁本层的 in_features。
        """
        if mask is None:
            # 表示上一步没真正剪枝，就不动
            return linear_layer

        old_in_dim = linear_layer.in_features
        num_to_keep = mask.sum().item()
        if num_to_keep == 0 or num_to_keep == old_in_dim:
            return linear_layer

        new_linear = nn.Linear(
            in_features=num_to_keep,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None
        ).to(linear_layer.weight.device).to(linear_layer.weight.dtype)

        with torch.no_grad():
            new_linear.weight.copy_(linear_layer.weight[:, mask])
            if linear_layer.bias is not None:
                new_linear.bias.copy_(linear_layer.bias)

        print(f"[Pruning] {linear_layer.__class__.__name__} in_features {old_in_dim} -> {num_to_keep}")
        return new_linear

    def _resize_dead_count(self, buffer_name: str, mask: torch.Tensor):
        """
        用于把对应 buffer (比如 dead_count_q1) 的形状也按照 mask 同步剪枝。
        同时保留历史统计中“存活神经元”的值。若想重置，可改为 zero_()。
        """
        if mask is None:
            return  # 无需剪

        old_buffer = getattr(self, buffer_name)
        new_buffer = old_buffer[mask].clone().to(old_buffer.device)
        self.register_buffer(buffer_name, new_buffer)

    def prune_neurons(self, step_count_threshold=100, zero_ratio_threshold=0.8):
        """
        每到一定训练步数执行一次：对 (qlinear1, qlinear2, klinear1, klinear2) 剪枝，
        并同步更新 dead_count_xx 的形状。
        """
        if self.forward_steps < step_count_threshold:
            return

        print(f"[prune_neurons] step={self.forward_steps}, zero_ratio_thresh={zero_ratio_threshold}")

        # ============== Q 路径 ==============
        # qlinear1
        self.qlinear1, mask_q1 = self._prune_one_linear(self.qlinear1, self.dead_count_q1, zero_ratio_threshold)
        # 同步剪 dead_count_q1
        self._resize_dead_count("dead_count_q1", mask_q1)
        # 更新 qlinear2.in_features
        self.qlinear2 = self._prune_next_layer_in(self.qlinear2, mask_q1)

        # qlinear2
        self.qlinear2, mask_q2 = self._prune_one_linear(self.qlinear2, self.dead_count_q2, zero_ratio_threshold)
        # 同步剪 dead_count_q2
        self._resize_dead_count("dead_count_q2", mask_q2)
        # 更新 qlinear3.in_features (此处不演示对 qlinear3.out_features 的剪枝)
        self.qlinear3 = self._prune_next_layer_in(self.qlinear3, mask_q2)

        # ============== K 路径 ==============
        self.klinear1, mask_k1 = self._prune_one_linear(self.klinear1, self.dead_count_k1, zero_ratio_threshold)
        self._resize_dead_count("dead_count_k1", mask_k1)
        self.klinear2 = self._prune_next_layer_in(self.klinear2, mask_k1)

        self.klinear2, mask_k2 = self._prune_one_linear(self.klinear2, self.dead_count_k2, zero_ratio_threshold)
        self._resize_dead_count("dead_count_k2", mask_k2)
        self.klinear3 = self._prune_next_layer_in(self.klinear3, mask_k2)

        # ========== 重置统计量，以便下一轮继续统计 =============
        self.forward_steps = 0
        self.total_count = 0
        # 这里也可以 choose to 保留 old_buffer ，但通常会 zero
        # 以便重新统计下一阶段激活情况
        self.dead_count_q1.zero_()
        self.dead_count_q2.zero_()
        self.dead_count_k1.zero_()
        self.dead_count_k2.zero_()

    def get_current_outdims(self):
        """
        返回当前 (q1_outdim, q2_outdim, k1_outdim, k2_outdim) 的实际 out_features。
        用于保存剪枝后的结构信息.
        """
        return {
            "q1_outdim": self.qlinear1.out_features,
            "q2_outdim": self.qlinear2.out_features,
            "k1_outdim": self.klinear1.out_features,
            "k2_outdim": self.klinear2.out_features
        }
        
class PrunableAttnPredictorInfer(nn.Module):
    """
    对 (qlinear1, qlinear2, klinear1, klinear2) 做结构化剪枝，
    并在剪完后同步更新 dead_count_xx 缓冲的形状，避免后续前向出现维度不匹配。
    """
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        q1_outdim=None, 
        q2_outdim=None, 
        k1_outdim=None, 
        k2_outdim=None, 
        n_head=32
    ):
        super(PrunableAttnPredictorInfer, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        
        # 若传入 q1_outdim 等为 None，表示尚未剪枝，初始化为默认形状
        q1_out = q1_outdim if q1_outdim is not None else hidden_dim
        q2_out = q2_outdim if q2_outdim is not None else (dim * 32)
        k1_out = k1_outdim if k1_outdim is not None else hidden_dim
        k2_out = k2_outdim if k2_outdim is not None else (dim * 32)

        # ============== Q 路径 ==============
        self.qlinear1 = nn.Linear(dim*64, q1_out)
        self.qlinear2 = nn.Linear(q1_out, q2_out)
        self.qlinear3 = nn.Linear(q2_out, dim)

        # ============== K 路径 ==============
        self.klinear1 = nn.Linear(dim*64, k1_out)
        self.klinear2 = nn.Linear(k1_out, k2_out)
        self.klinear3 = nn.Linear(k2_out, dim)

    def forward(self, hidden_states):
        """
        hidden_states 形状: [batch_size, seq_len, hidden_dim], 
        其中 hidden_dim = self.dim * self.n_head (仅示例).
        """
        bsz, q_len, _ = hidden_states.size()

        # 你的 reshape 逻辑 (示例)
        hidden_states = hidden_states.view(bsz, q_len, self.n_head, -1).transpose(1, 2)
        hidden_states = hidden_states.reshape(bsz, self.n_head, q_len // 64, -1)

        # ============ Q 路径 ============
        qx1 = self.qlinear1(hidden_states)
        qx1 = F.relu(qx1)

        qx2 = self.qlinear2(qx1)
        qx2 = F.relu(qx2)

        qx = self.qlinear3(qx2)

        # ============ K 路径 ============
        kx1 = self.klinear1(hidden_states)
        kx1 = F.relu(kx1)

        kx2 = self.klinear2(kx1)
        kx2 = F.relu(kx2)

        kx = self.klinear3(kx2)

        # 计算 attn
        attn = torch.matmul(qx, kx.transpose(-1, -2))
        attn = attn.sum(dim=1)

        return attn
