"""Trainable KV-binding adapter read out at attention o_proj output.

Learns three projections on frozen-base hidden states:
    W_K : d_model -> d_addr   (write-side address key)
    W_Q : d_model -> d_addr   (query address)
    W_V : d_model -> d_model   (write-side value, lives in o_proj OUTPUT / residual space)

An episode compiles N facts into a single associative map via a MEMIT/ridge solve
    DeltaO = V^T K (K^T K + lambda I)^-1        # [d_model, d_addr]
and reads it with the query
    delta_y = DeltaO q                          # [d_model]  -> added to o_proj output.

This replaces DeltaMem's online delta-rule + rank-8 bottleneck: d_addr is free (64/128/...),
the value is produced directly in residual space, and the readout is injected at o_proj output
(the causal fact-readout site found by the earlier probe). Ridge solve is done in fp32.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class OProjKVBinding(nn.Module):
    def __init__(self, d_model: int, d_addr: int = 128, target_layer: int = 24,
                 lambda_ridge: float = 1e-2, beta: float = 1.0,
                 center_keys: bool = True, normalize_kq: bool = True,
                 tie_kq: bool = False, learn_beta: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_addr = d_addr
        self.target_layer = target_layer
        self.lambda_ridge = lambda_ridge
        self.center_keys = center_keys
        self.normalize_kq = normalize_kq
        self.tie_kq = tie_kq
        self.W_K = nn.Linear(d_model, d_addr, bias=False)
        self.W_Q = self.W_K if tie_kq else nn.Linear(d_model, d_addr, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.beta_max = 2.0
        self._reset()
        if learn_beta:
            # bounded: beta = beta_max * sigmoid(beta_raw); init so beta == `beta`
            import math as _m
            r0 = _m.log(beta / max(self.beta_max - beta, 1e-4))
            self.beta_raw = nn.Parameter(torch.tensor(float(r0)))
            self.log_beta = None
        else:
            self.register_buffer("beta_const", torch.tensor(float(beta)))
            self.log_beta = None; self.beta_raw = None

    def _reset(self):
        # small init so tanh stays in linear regime; V near-identity-scale not needed (supervised)
        nn.init.normal_(self.W_K.weight, std=self.d_model ** -0.5)
        if not self.tie_kq:
            nn.init.normal_(self.W_Q.weight, std=self.d_model ** -0.5)
        nn.init.normal_(self.W_V.weight, std=self.d_model ** -0.5)

    @property
    def beta(self):
        if self.beta_raw is not None:
            return self.beta_max * torch.sigmoid(self.beta_raw)
        return self.beta_const

    def encode_key(self, h):        # h: [..., d_model] -> [..., d_addr]
        k = self.W_K(h.to(self.W_K.weight.dtype))
        if self.normalize_kq:
            k = F.normalize(torch.tanh(k), dim=-1, eps=1e-6)
        return k

    def encode_query(self, h):
        q = self.W_Q(h.to(self.W_Q.weight.dtype))
        if self.normalize_kq:
            q = F.normalize(torch.tanh(q), dim=-1, eps=1e-6)
        return q

    def encode_value(self, h):      # -> [..., d_model]
        return self.W_V(h.to(self.W_V.weight.dtype))

    def solve(self, K, V, lam=None):
        """K: [B,N,d_addr]  V: [B,N,d_model] -> DeltaO: [B,d_model,d_addr] (fp32)."""
        lam = self.lambda_ridge if lam is None else lam
        Kf, Vf = K.float(), V.float()
        I = torch.eye(self.d_addr, device=K.device, dtype=torch.float32)
        KtK = torch.einsum("bnd,bne->bde", Kf, Kf) + lam * I          # [B,d_addr,d_addr]
        VtK = torch.einsum("bnm,bnd->bmd", Vf, Kf)                    # [B,d_model,d_addr]
        DeltaO = torch.linalg.solve(KtK.transpose(1, 2), VtK.transpose(1, 2)).transpose(1, 2)
        return DeltaO                                                 # [B,d_model,d_addr]

    def forward(self, H_write_key, H_write_value, H_query, lam=None):
        """H_*: [B,*,d_model]. Returns dict with delta_y [B,M,d_model] and internals."""
        K = self.encode_key(H_write_key)         # [B,N,d_addr]
        Q = self.encode_query(H_query)           # [B,M,d_addr]
        V = self.encode_value(H_write_value)     # [B,N,d_model]
        mu = None
        if self.center_keys:
            mu = K.mean(dim=1, keepdim=True)     # shared write-side mean
            K = K - mu
            Q = Q - mu
        DeltaO = self.solve(K, V, lam)                                   # [B,d_model,d_addr]
        delta_y = torch.einsum("bqe,bme->bqm", Q.float(), DeltaO)        # [B,M,d_model]
        recon = torch.einsum("bne,bme->bnm", K.float(), DeltaO)          # DeltaO k_i [B,N,d_model]
        return {"delta_y": delta_y, "K": K, "Q": Q, "V": V, "DeltaO": DeltaO,
                "recon": recon, "mu": mu, "beta": self.beta}
