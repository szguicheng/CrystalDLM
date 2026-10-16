"""Matched models isolating multiplicity-partition probability factorization."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize_scores(coarse, fine, local_partition, valid, mode):
    groups = torch.arange(coarse.shape[-1], device=fine.device)
    membership = (local_partition.unsqueeze(-2) == groups.view(*([1] * (fine.ndim - 1)), -1, 1)) & valid.unsqueeze(-2)
    available = membership.any(-1)
    coarse = coarse.masked_fill(~available, -1e4)
    index = local_partition.clamp_min(0)
    if mode == 'flat':
        scores = fine + coarse.gather(-1, index)
    elif mode == 'degeneracy':
        log_size = membership.sum(-1).clamp_min(1).float().log()
        scores = fine + (coarse - log_size).gather(-1, index)
    elif mode == 'hierarchical':
        within = fine.unsqueeze(-2).masked_fill(~membership, -1e4).logsumexp(-1)
        scores = fine + (F.log_softmax(coarse, -1) - within).gather(-1, index)
    else:
        raise ValueError(mode)
    return F.log_softmax(scores.masked_fill(~valid, -1e4), -1)


class MultiplicityModel(nn.Module):
    def __init__(self, data, mode, width=96):
        super().__init__()
        self.mode, self.width = mode, width
        for name in ['profile_counts', 'profile_wp_ids', 'profile_sg', 'wp_metadata', 'context_candidates',
                     'primitive_partition_counts', 'profile_partition_id', 'context_partition_ids',
                     'candidate_partition_local', 'centering_factors']:
            self.register_buffer(name, data[name])
        self.element = nn.Embedding(119, width, padding_idx=0)
        self.amount = nn.Embedding(data['max_atoms'] + 1, width, padding_idx=0)
        self.primitive_amount = nn.Embedding(data['max_atoms'] + 1, width, padding_idx=0)
        self.sg = nn.Embedding(231, width, padding_idx=0)
        self.wp = nn.Embedding(len(data['wp_metadata']), width, padding_idx=0)
        self.mu = nn.Embedding(49, width, padding_idx=0)
        self.dof = nn.Embedding(4, width)
        largest = max(int(data['profile_counts'].max()), int(data['primitive_partition_counts'].max()))
        self.count = nn.Embedding(largest + 1, width, padding_idx=0)
        self.fine_projection = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.coarse_projection = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.state_projection = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.mask = nn.Parameter(torch.zeros(width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                    activation='gelu', batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.fine_query = nn.Linear(width, width)
        self.coarse_query = nn.Linear(width, width)

    def encode_profiles(self, ids):
        counts = self.profile_counts[ids]
        row, col = (counts > 0).nonzero(as_tuple=True)
        wpids = self.profile_wp_ids[ids][row, col]
        metadata = self.wp_metadata[wpids]
        centers = self.centering_factors[self.profile_sg[ids[row]]]
        primitive_mu = metadata[:, 0] // centers
        fields = (self.wp(wpids) + self.mu(primitive_mu) + self.dof(metadata[:, 1])) * self.count(counts[row, col])
        values = torch.zeros(len(ids), self.width, device=fields.device).index_add(0, row, fields)
        return self.fine_projection(values)

    def encode_partitions(self):
        row, mu = (self.primitive_partition_counts > 0).nonzero(as_tuple=True)
        count = self.primitive_partition_counts[row, mu]
        fields = self.mu(mu) * self.count(count)
        values = torch.zeros(len(self.primitive_partition_counts), self.width, device=fields.device).index_add(0, row, fields)
        return self.coarse_projection(values)

    def forward(self, batch, hidden):
        candidates = self.context_candidates[batch['context']]
        valid = candidates >= 0
        ids, inverse = torch.unique(torch.cat((candidates.clamp_min(0).reshape(-1), batch['profile_id'].reshape(-1))), return_inverse=True)
        encoded = self.encode_profiles(ids)
        split = candidates.numel()
        fine_candidates = encoded[inverse[:split]].reshape(*candidates.shape, self.width)
        observed_fine = encoded[inverse[split:]].reshape(*batch['profile_id'].shape, self.width)
        partition_vectors = self.encode_partitions()
        parent_ids = self.context_partition_ids[batch['context']]
        coarse_candidates = partition_vectors[parent_ids.clamp_min(0)]
        observed_coarse = partition_vectors[self.profile_partition_id[batch['profile_id']]]
        state = self.state_projection(torch.cat((observed_fine, observed_coarse), -1))
        state = torch.where(hidden.unsqueeze(-1), self.mask, state)
        center = self.centering_factors[batch['sg']].unsqueeze(-1)
        h = (self.element(batch['Z']) + self.amount(batch['N']) + self.primitive_amount(batch['N'] // center)
             + self.sg(batch['sg']).unsqueeze(1) + state)
        h = self.trunk(h, src_key_padding_mask=batch['Z'] == 0)
        fine = torch.einsum('bed,becd->bec', self.fine_query(h), fine_candidates) / math.sqrt(self.width)
        coarse = torch.einsum('bed,bekd->bek', self.coarse_query(h), coarse_candidates) / math.sqrt(self.width)
        scores = normalize_scores(coarse, fine, self.candidate_partition_local[batch['context']], valid, self.mode)
        return scores, valid
