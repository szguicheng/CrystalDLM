"""Matched site-type inventory / WP-record encoders for masked profile prediction.

This isolates count aggregation, not a complete inventory-first diffusion model.
Both variants retain the same fine WP branch and use identical parameters.
"""
import torch
from torch import nn
from multiplicity_unaggregated import UnaggregatedModel


class SiteInventoryModel(UnaggregatedModel):
    def __init__(self, data, mode, width=96):
        assert mode in ('aggregate', 'records')
        super().__init__(data, 'flat', width)
        self.aggregation_mode = mode
        self.site_type = nn.Embedding(33, width, padding_idx=0)
        self.register_buffer('wp_site_type', data['wp_site_type'])

    def encode_profiles(self, ids):
        counts = self.profile_counts[ids]
        row, col = (counts > 0).nonzero(as_tuple=True)
        wpids = self.profile_wp_ids[ids][row, col]
        metadata = self.wp_metadata[wpids]
        centers = self.centering_factors[self.profile_sg[ids[row]]]
        fields = (self.wp(wpids) + self.mu(metadata[:, 0] // centers)
                  + self.dof(metadata[:, 1]) + self.site_type(self.wp_site_type[wpids]))
        fields = fields * self.count(counts[row, col])
        values = torch.zeros(len(ids), self.width, device=fields.device).index_add(0, row, fields)
        return self.fine_projection(values)

    def encode_unaggregated(self, ids):
        counts = self.profile_counts[ids]
        row, col = (counts > 0).nonzero(as_tuple=True)
        wpids = self.profile_wp_ids[ids][row, col]
        types = self.wp_site_type[wpids]
        amounts = counts[row, col]
        if self.aggregation_mode == 'aggregate':
            inventory = torch.zeros(len(ids) * 33, device=counts.device, dtype=counts.dtype)
            inventory.index_add_(0, row * 33 + types, amounts)
            inventory = inventory.reshape(len(ids), 33)
            row, types = (inventory > 0).nonzero(as_tuple=True)
            amounts = inventory[row, types]
        fields = self.site_type(types) * self.count(amounts)
        values = torch.zeros(len(ids), self.width, device=fields.device).index_add(0, row, fields)
        return self.coarse_projection(values)
