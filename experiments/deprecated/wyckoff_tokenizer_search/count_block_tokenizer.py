"""Portable, lossless Wyckoff-count block codec; Python standard library only.

A token is a tuple of (Wyckoff letter, orbit count) records for one element.
Space group and element identity are external headers. Geometry is not encoded.
"""
from collections import Counter
import json
from pathlib import Path


def records(value):
    return tuple(sorted((letter, int(count)) for letter, count in value))


class CountBlockTokenizer:
    def __init__(self, path):
        self.spec = json.loads(Path(path).read_text())
        self.rules = {int(g): [tuple(records(part) for part in rule) for rule in rules]
                      for g, rules in self.spec['rules_by_group'].items()}
        self.order = {int(g): {records(item['records']): item['id'] for item in items}
                      for g, items in self.spec['word_order_by_group'].items()}
        self.positions = {int(g): value for g, value in self.spec['positions'].items()}

    def encode(self, space_group, profile):
        """Encode {letter: independent_orbit_count}, with typed RLE fallback."""
        assert all(isinstance(n, int) and n >= 0 for n in profile.values())
        bag = {((letter, n),) for letter, n in profile.items() if n}
        for left, right, merged in self.rules.get(space_group, ()):
            if left in bag and right in bag:
                bag.remove(left); bag.remove(right); bag.add(merged)
        order = self.order.get(space_group, {})
        # Preserve the frozen model's canonical order for its known vocabulary.
        # Unseen singleton count records remain representable without new rules.
        return tuple(sorted(bag, key=lambda token: (order.get(token, 10**12), token)))

    @staticmethod
    def decode(tokens):
        profile = Counter()
        for token in tokens:
            for letter, count in token:
                profile[letter] += count
        return dict(sorted(profile.items()))

    def atom_count(self, space_group, tokens):
        return sum(self.positions[space_group][letter]['multiplicity'] * n
                   for letter, n in self.decode(tokens).items())
