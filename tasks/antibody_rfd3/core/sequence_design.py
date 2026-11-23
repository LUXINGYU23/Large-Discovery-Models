"""Agent-directed sequence proposals with explicit, reproducible constraints."""
from itertools import combinations
import math
import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def validate_sequence(sequence):
    if not isinstance(sequence, str) or len(sequence) != 11 or set(sequence) - set(AMINO_ACIDS):
        raise ValueError("Sequences must contain exactly 11 uppercase canonical amino acids")
    return sequence


def mutation_options(seed_sequence, mutable_positions, min_mutations, max_mutations, allowed_amino_acids):
    validate_sequence(seed_sequence)
    positions = list(range(1, 12)) if mutable_positions is None else mutable_positions
    if (not isinstance(positions, list) or not positions or
            any(type(p) is not int or not 1 <= p <= 11 for p in positions) or len(set(positions)) != len(positions)):
        raise ValueError("mutable_positions must be distinct 1-based H3 positions in [1,11]")
    if (type(min_mutations) is not int or type(max_mutations) is not int
            or not 1 <= min_mutations <= max_mutations <= len(positions)):
        raise ValueError("Mutation counts must satisfy 1 <= min_mutations <= max_mutations <= mutable position count")
    allowed = {} if allowed_amino_acids is None else allowed_amino_acids
    if not isinstance(allowed, dict) or any(p not in {str(i) for i in positions} for p in allowed):
        raise ValueError("allowed_amino_acids keys must name mutable positions")
    options = {}
    for p in positions:
        alphabet = allowed.get(str(p), AMINO_ACIDS)
        if (not isinstance(alphabet, str) or not alphabet or set(alphabet) - set(AMINO_ACIDS)
                or len(set(alphabet)) != len(alphabet)):
            raise ValueError("Each allowed amino acid alphabet must be nonempty, uppercase and unique")
        options[p - 1] = tuple(a for a in alphabet if a != seed_sequence[p - 1])
    groups = [(subset, math.prod(len(options[p]) for p in subset))
              for n in range(min_mutations, max_mutations + 1) for subset in combinations(options, n)]
    return options, [(subset, size) for subset, size in groups if size]


def mutate_sequences(seed_sequence, count, options, groups, seed):
    capacity = sum(size for _, size in groups)
    if count > capacity:
        raise ValueError(f"Requested {count} unique mutations but constraints allow only {capacity}")
    # Sample integer ranks without rejection loops even for tiny constrained spaces.
    ranks = random.Random(seed).sample(range(capacity), count)
    sequences = []
    for rank in ranks:
        for subset, size in groups:
            if rank < size:
                break
            rank -= size
        letters = list(seed_sequence)
        for position in reversed(subset):
            rank, index = divmod(rank, len(options[position]))
            letters[position] = options[position][index]
        sequences.append("".join(letters))
    return sequences
