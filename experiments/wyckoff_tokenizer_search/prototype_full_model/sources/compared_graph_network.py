"""Add an occurrence-level ASU control; preserve the frozen network constructor."""
import torch
from orbit_graph_network import OrbitGraphNetwork


class ComparedGraphNetwork(OrbitGraphNetwork):
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
        elif mode == 'asu':
            src, dst, graph = d['ausrc'], d['audst'], d['asu_graph']
            delta = torch.einsum('nij,nj->ni', d['auA'], q[dst]) + d['aub'] - q[src]
            vector = torch.einsum('ni,nij->nj', delta, d['lattice'][graph[src]])
            vector_sum = vector
            counts = torch.ones(len(src), device=q.device, dtype=q.dtype)
            h = self.embedding(d['Z'])
            weights = d['multiplicity']
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
            weighted = m * counts[:, None] if mode == 'orbit' else m
            aggregate = torch.zeros_like(h).index_add(0, src, weighted) / degree[:, None]
            h = h + update(torch.cat((h, aggregate), -1))
        scalar = self.vector_weight(torch.cat((h[src], h[dst], radial), -1))
        force = torch.zeros(len(h), 3, device=q.device, dtype=q.dtype).index_add(0, src, scalar * vector_sum) / degree[:, None]
        ng = len(d['lattice'])
        total = torch.zeros(ng, device=q.device, dtype=q.dtype).index_add(0, graph, weights)
        pooled = torch.zeros(ng, h.shape[-1], device=q.device, dtype=q.dtype).index_add(0, graph, h * weights[:, None]) / total[:, None]
        vector_norm = torch.zeros(ng, device=q.device, dtype=q.dtype).index_add(0, graph, weights * force.square().sum(-1)) / total
        return pooled, force, vector_norm
