"""Shared radial message network with a vector output, full/quotient paths.

The quotient path evaluates each edge class once and sums its directions using
precomputed affine coefficients. It never expands full-cell edges in forward.
"""
import torch
from torch import nn


class OrbitGraphNetwork(nn.Module):
    def __init__(self, width=64, layers=3):
        super().__init__()
        self.embedding = nn.Embedding(119, width)
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(width * 2 + 16, width), nn.SiLU(),
                                                    nn.Linear(width, width), nn.SiLU()) for _ in range(layers)])
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(),
                                                   nn.Linear(width, width)) for _ in range(layers)])
        self.vector_weight = nn.Sequential(nn.Linear(width * 2 + 16, width), nn.SiLU(), nn.Linear(width, 1))
        self.register_buffer('centers', torch.linspace(0., 6., 16))

    def forward(self, d, u, mode):
        q = d['q'] + torch.einsum('nij,nj->ni', d['projection'], u)
        if mode == 'full':
            frac = torch.einsum('nij,nj->ni', d['A'], q[d['orbit']]) + d['b']
            src, dst, graph = d['src'], d['dst'], d['full_graph']
            vector = torch.einsum('ni,nij->nj', frac[dst] + d['image'] - frac[src], d['lattice'][graph[src]])
            vector_sum = vector
            counts = torch.ones(len(src), device=q.device, dtype=q.dtype)
            h = self.embedding(d['Z'][d['orbit']])
            weights = torch.ones(len(h), device=q.device, dtype=q.dtype)
        elif mode == 'orbit':
            src, dst, graph = d['cs'], d['ct'], d['asu_graph']
            delta = torch.einsum('nij,nj->ni', d['cA'], q[dst]) + d['cb'] - q[src]
            vector = torch.einsum('ni,nij->nj', delta, d['lattice'][graph[src]])
            counts = d['count']
            delta_sum = torch.einsum('nij,nj->ni', d['sumA'], q[dst]) + d['sumb'] - counts[:, None] * q[src]
            vector_sum = torch.einsum('ni,nij->nj', delta_sum, d['lattice'][graph[src]])
            h = self.embedding(d['Z'])
            weights = d['multiplicity']
        else:
            raise ValueError(mode)
        distances = vector.square().sum(-1).clamp_min(1e-16).sqrt()
        radial = torch.exp(-((distances[:, None] - self.centers) / .4).square())
        degree = torch.zeros(len(h), device=q.device, dtype=q.dtype).index_add(0, src, counts).clamp_min(1)
        for message, update in zip(self.messages, self.updates):
            m = message(torch.cat((h[src], h[dst], radial), -1))
            aggregate = torch.zeros_like(h).index_add(0, src, m * counts[:, None]) / degree[:, None]
            h = h + update(torch.cat((h, aggregate), -1))
        scalar = self.vector_weight(torch.cat((h[src], h[dst], radial), -1))
        force = torch.zeros(len(h), 3, device=q.device, dtype=q.dtype).index_add(0, src, scalar * vector_sum) / degree[:, None]
        ng = len(d['lattice'])
        total = torch.zeros(ng, device=q.device, dtype=q.dtype).index_add(0, graph, weights)
        pooled = torch.zeros(ng, h.shape[-1], device=q.device, dtype=q.dtype).index_add(0, graph, h * weights[:, None]) / total[:, None]
        vector_norm = torch.zeros(ng, device=q.device, dtype=q.dtype).index_add(0, graph, weights * force.square().sum(-1)) / total
        return pooled, force, vector_norm
