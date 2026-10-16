"""Matched word encoder for primitive, run-length and learned set-merge units."""
import torch
from torch import nn
from train_profile_model import ProfileModel


class WordModel(ProfileModel):
    def __init__(self, data, variant, width=128, layers=3):
        super().__init__(data, 'factorized', width, layers)
        self.word_variant = variant
        self.register_buffer('word_counts', data['word_counts'])
        self.register_buffer('word_wp_ids', data['word_wp_ids'])
        self.register_buffer('profile_words', data['profile_words'][variant])
        self.word_projection = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def encode_profiles(self, profile_ids):
        words = self.profile_words[profile_ids]
        ids, inverse = torch.unique(words.reshape(-1), return_inverse=True)
        counts = self.word_counts[ids]
        wp_ids = self.word_wp_ids[ids]
        metadata = self.wp_metadata[wp_ids]
        features = self.wp(wp_ids) + self.multiplicity(metadata[..., 0]) + self.dof(metadata[..., 1])
        contributions = features * self.occupancy(counts)
        vectors = self.word_projection((contributions * (counts > 0).unsqueeze(-1)).sum(-2))
        vectors = vectors * (ids > 0).unsqueeze(-1)
        return vectors[inverse].reshape(*words.shape, self.width).sum(-2)
