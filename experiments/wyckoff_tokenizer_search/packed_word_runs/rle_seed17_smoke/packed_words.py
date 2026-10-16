"""Condition-derived slot packing, with unchanged frozen word-model parameters."""
import math
import torch
from direct_words import DirectWordModel


class PackedWordModel(DirectWordModel):
    def __init__(self, data, variant):
        super().__init__(data, variant, compact=True)
        maxima = []
        for row in data['context_candidates']:
            ids = row[row >= 0]
            maxima.append(int((self.profile_words[ids] > 0).sum(-1).max()))
        self.register_buffer('context_max_words', torch.tensor(maxima))

    def valid_slots(self, batch):
        positions = torch.arange(self.length, device=batch['Z'].device)
        return ((batch['Z'] > 0).unsqueeze(-1) &
                (positions < self.context_max_words[batch['context']].unsqueeze(-1)))

    def encode_words(self, ids):
        # Compute only occupied records, rather than 27 padded WP feature rows.
        counts = self.word_counts[ids]
        rows, columns = (counts > 0).nonzero(as_tuple=True)
        wpids = self.word_wp_ids[ids][rows, columns]
        n = counts[rows, columns]
        metadata = self.wp_metadata[wpids]
        fields = self.wp(wpids) + self.multiplicity(metadata[:, 0]) + self.dof(metadata[:, 1])
        fields = fields * self.occupancy(n)
        summed = torch.zeros(len(ids), self.width, device=fields.device, dtype=fields.dtype).index_add(0, rows, fields)
        vectors = self.word_projection(summed)
        return torch.where((ids == 0).unsqueeze(-1), self.pad, vectors)

    def forward(self, batch, words, hidden, packed=True):
        allowed_words = self.context_words[batch['context']]
        ids, inverse = torch.unique(torch.cat((allowed_words.clamp_min(0).reshape(-1), words.reshape(-1))), return_inverse=True)
        vectors = self.encode_words(ids)
        split = allowed_words.numel()
        output_vectors = vectors[inverse[:split]].reshape(*allowed_words.shape, self.width)
        observed = vectors[inverse[split:]].reshape(*words.shape, self.width)
        state = torch.where(hidden.unsqueeze(-1), self.mask, observed)
        condition = self.element(batch['Z']) + self.amount(batch['N']) + self.sg(batch['sg']).unsqueeze(1)
        positions = self.slot(torch.arange(self.length, device=words.device))
        x = condition.unsqueeze(2) + positions + self.state_norm(state)
        real = self.valid_slots(batch)
        flat = x.flatten(1, 2)
        valid = real.flatten(1, 2)
        batch_size, dense_length, width = flat.shape
        if packed:
            length = int(valid.sum(-1).max())
            indices = torch.arange(dense_length, device=x.device).expand(batch_size, -1)
            indices = indices.masked_fill(~valid, dense_length).sort(-1).values[:, :length]
            padded = torch.cat((flat, torch.zeros_like(flat[:, :1])), dim=1)
            compact = padded.gather(1, indices.unsqueeze(-1).expand(-1, -1, width))
            h = self.head(self.trunk(compact, src_key_padding_mask=indices == dense_length))
            expanded = torch.zeros_like(padded).scatter(1, indices.unsqueeze(-1).expand(-1, -1, width), h)
            h = expanded[:, :dense_length]
            self.last_packed_length = length
        else:
            h = self.head(self.trunk(flat, src_key_padding_mask=~valid))
        self.last_dense_length = dense_length
        h = h.reshape(*words.shape, self.width)
        logits = torch.einsum('beld,bekd->belk', h, output_vectors) / math.sqrt(self.width)
        logits = logits.masked_fill((allowed_words < 0).unsqueeze(2), -1e4)
        # These positions are known PAD for every candidate under the input G,N.
        fixed_pad = torch.full_like(logits, -1e4)
        fixed_pad[..., 0] = 0.
        return torch.where(real.unsqueeze(-1), logits, fixed_pad)
