import torch
import torch.nn as nn
import math

def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    M = M / reg  # Regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

def get_matching_probs(S, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()

    norm = -torch.tensor(math.log(n), device=S.device)
    log_a, log_b = norm.expand(m).contiguous(), norm.expand(n).contiguous()
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)

    log_P = log_otp_solver(
        log_a,
        log_b,
        S,
        num_iters=num_iters,
        reg=reg
    )

    return log_P - norm

class image_aggregation(nn.Module):

    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 temperature=1.0
                 ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )

        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)

        p = get_matching_probs(p, num_iters=3)

        p = torch.exp(p / self.temperature)+torch.exp(p)  # 调整分配概率的平滑程度

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = (f * p).sum(dim=-1)

        return nn.functional.normalize(f.flatten(1), p=2, dim=-1)
