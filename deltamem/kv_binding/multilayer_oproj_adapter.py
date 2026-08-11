"""Multi-layer late-o_proj KV-binding adapter.

Diagnosis that motivates this (oracle patching on synthetic single-chunk):
  - single L24 o_proj injection: too weak (0/10 generation).
  - down_proj: harmful.  early layers: harmful.
  - LATE-layer multi-o_proj + full-suffix (all task positions) injection: first-token rank ~1,
    answer hit 7-8/10, ~= oracle-context ceiling.

Architecture: shared W_K/W_Q, per-layer W_V_l -> per-layer DeltaO_l (batch ridge), per-layer bounded
beta_l (small init).  Memory is fixed batch-ridge (NOT sequential).  Injection is content-addressed at
ALL task-suffix positions (parallel add, not sequential):
    o_proj_out_l[:, s, :] += beta_l * (DeltaO_l @ q_l,s)     for all suffix positions s.
No down_proj, no early layers, base frozen, o_proj.weight untouched.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerOProjKVBinding(nn.Module):
    def __init__(self, d_model: int, layers, d_addr: int = 128, lambda_ridge: float = 1e-2,
                 beta_init: float = 0.05, beta_max: float = 0.5, center_keys: bool = False,
                 tie_kq: bool = True, normalize_kq: bool = True):
        super().__init__()
        self.d_model = d_model
        self.layers = list(layers)
        self.d_addr = d_addr
        self.lambda_ridge = lambda_ridge
        self.beta_max = beta_max
        self.center_keys = center_keys
        self.normalize_kq = normalize_kq
        self.W_K = nn.Linear(d_model, d_addr, bias=False)              # shared retrieval/binding key
        self.W_Q = self.W_K if tie_kq else nn.Linear(d_model, d_addr, bias=False)
        self.W_V = nn.ModuleDict({str(l): nn.Linear(d_model, d_model, bias=False) for l in self.layers})
        r0 = math.log(beta_init / max(beta_max - beta_init, 1e-4))
        self.beta_raw = nn.ParameterDict({str(l): nn.Parameter(torch.tensor(float(r0))) for l in self.layers})
        nn.init.normal_(self.W_K.weight, std=d_model ** -0.5)
        if not tie_kq:
            nn.init.normal_(self.W_Q.weight, std=d_model ** -0.5)
        for l in self.layers:
            nn.init.normal_(self.W_V[str(l)].weight, std=d_model ** -0.5)

    def beta(self, l):
        return self.beta_max * torch.sigmoid(self.beta_raw[str(l)])

    def encode_key(self, h):
        k = self.W_K(h.to(self.W_K.weight.dtype))
        return F.normalize(torch.tanh(k), dim=-1, eps=1e-6) if self.normalize_kq else k

    def encode_query(self, h):
        q = self.W_Q(h.to(self.W_Q.weight.dtype))
        return F.normalize(torch.tanh(q), dim=-1, eps=1e-6) if self.normalize_kq else q

    def value(self, l, h):
        return self.W_V[str(l)](h.to(self.W_V[str(l)].weight.dtype))

    def solve(self, K, V, lam=None):
        """K [N,d_addr], V [N,d_model] -> DeltaO [d_model,d_addr] (fp32)."""
        lam = self.lambda_ridge if lam is None else lam
        Kf = torch.nan_to_num(K.float()); Vf = torch.nan_to_num(V.float())   # guard degenerate/empty chunks
        I = torch.eye(self.d_addr, device=K.device, dtype=torch.float32)
        KtK = Kf.t() @ Kf + lam * I
        VtK = Vf.t() @ Kf
        try:
            return torch.linalg.solve(KtK.t(), VtK.t()).t()           # [d_model,d_addr]
        except Exception:
            return (VtK @ torch.linalg.pinv(KtK))                     # fallback

    def build_memory(self, chunk_h: dict):
        """chunk_h: {l: [N,d_model]} layer-l hidden of the selected chunks.
        Returns {l: (DeltaO_l [d_model,d_addr], mu_l [d_addr] or None)}."""
        mem = {}
        for l in self.layers:
            h = chunk_h[l]
            K = self.encode_key(h); V = self.value(l, h)
            mu = K.mean(0, keepdim=True) if (self.center_keys and h.shape[0] > 1) else None
            Kc = K - mu if mu is not None else K
            mem[l] = (self.solve(Kc, V), (mu[0] if mu is not None else None))
        return mem
