"""WP-record control: embed each count before summing equal-multiplicity records.

Inherits the exact parameter constructor of the frozen multiplicity model.
"""
import math
import torch
from torch.nn import functional as F
from multiplicity_model import MultiplicityModel


class UnaggregatedModel(MultiplicityModel):
    def encode_unaggregated(self, ids):
        counts = self.profile_counts[ids]
        row, col = (counts > 0).nonzero(as_tuple=True)
        wpids = self.profile_wp_ids[ids][row, col]
        centers = self.centering_factors[self.profile_sg[ids[row]]]
        mu = self.wp_metadata[wpids, 0] // centers
        fields = self.mu(mu) * self.count(counts[row, col])
        summed = torch.zeros(len(ids), self.width, device=fields.device).index_add(0, row, fields)
        return self.coarse_projection(summed)

    def forward(self, batch, hidden):
        candidates = self.context_candidates[batch['context']]
        valid = candidates >= 0
        ids, inverse = torch.unique(torch.cat((candidates.clamp_min(0).reshape(-1), batch['profile_id'].reshape(-1))), return_inverse=True)
        fine_vectors = self.encode_profiles(ids)
        coarse_vectors = self.encode_unaggregated(ids)
        split = candidates.numel()
        fine_candidates = fine_vectors[inverse[:split]].reshape(*candidates.shape, self.width)
        coarse_candidates = coarse_vectors[inverse[:split]].reshape(*candidates.shape, self.width)
        observed_fine = fine_vectors[inverse[split:]].reshape(*batch['profile_id'].shape, self.width)
        observed_coarse = coarse_vectors[inverse[split:]].reshape(*batch['profile_id'].shape, self.width)
        state = self.state_projection(torch.cat((observed_fine, observed_coarse), -1))
        state = torch.where(hidden.unsqueeze(-1), self.mask, state)
        centers = self.centering_factors[batch['sg']].unsqueeze(-1)
        h = (self.element(batch['Z']) + self.amount(batch['N']) + self.primitive_amount(batch['N'] // centers)
             + self.sg(batch['sg']).unsqueeze(1) + state)
        h = self.trunk(h, src_key_padding_mask=batch['Z'] == 0)
        fine = torch.einsum('bed,becd->bec', self.fine_query(h), fine_candidates) / math.sqrt(self.width)
        coarse = torch.einsum('bed,becd->bec', self.coarse_query(h), coarse_candidates) / math.sqrt(self.width)
        scores = fine + coarse
        return F.log_softmax(scores.masked_fill(~valid, -1e4), -1), valid
