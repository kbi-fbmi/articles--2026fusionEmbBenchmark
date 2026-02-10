#!/usr/bin/env python3
"""Script for augmenting DNA sequences with random base mutations.

Loads fusionai_test_sim.txt, performs mutations, and saves to a new file.
"""

import argparse
import random
import sys
from pathlib import Path

# Add path to fmlib module
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_fusions_from_fusionaitxt(file_path, fused_lambda=None):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            columns = line.strip().split("\t")
            if len(columns) == 11:
                entry = {
                    "gene1": columns[0],
                    "chr1": columns[1],
                    "pos1": int(columns[2]),
                    "strand1": columns[3],
                    "gene2": columns[4],
                    "chr2": columns[5],
                    "pos2": int(columns[6]),
                    "strand2": columns[7],
                    "sequence1": columns[8],
                    "sequence2": columns[9],
                    "target": columns[10],
                }
                if "N" in entry["sequence1"] or "N" in entry["sequence2"]:
                    continue
                data.append(entry)
    return data


def mutate_sequence(sequence: str, mutation_rate: float) -> str:
    """Mutate DNA sequence with given mutation probability.

    Parameters
    ----------
    sequence : str
        DNA sequence (ACGT)
    mutation_rate : float
        Percentage of bases to mutate (0-100)

    Returns
    -------
    str
        Mutated sequence

    """
    bases = ["A", "C", "G", "T"]
    sequence_list = list(sequence)
    num_mutations = int(len(sequence) * (mutation_rate / 100.0))

    # Randomly select positions to mutate
    positions_to_mutate = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))

    for pos in positions_to_mutate:
        original_base = sequence_list[pos]
        # Select a different base than the original
        possible_bases = [b for b in bases if b != original_base]
        sequence_list[pos] = random.choice(possible_bases)

    return "".join(sequence_list)


def augment_fusions(
    input_path: str,
    output_path: str,
    mutation_rate: float,
    seed: int | None = None,
) -> None:
    """Load fusion data, mutate sequences, and save to a new file.

    Parameters
    ----------
    input_path : str
        Path to input file
    output_path : str
        Path to output file
    mutation_rate : float
        Percentage of bases to mutate (0-100)
    seed : int, optional
        Seed for random number generator

    """
    if seed is not None:
        random.seed(seed)

    # Load data
    print(f"Loading data from: {input_path}")
    data = load_fusions_from_fusionaitxt(input_path)
    print(f"Loaded {len(data)} samples")

    # Augment data
    print(f"Performing mutations with {mutation_rate}% mutation rate...")
    augmented_data = []
    for entry in data:
        augmented_entry = entry.copy()
        augmented_entry["sequence1"] = mutate_sequence(entry["sequence1"], mutation_rate)
        augmented_entry["sequence2"] = mutate_sequence(entry["sequence2"], mutation_rate)
        augmented_data.append(augmented_entry)

    # Save augmented data
    print(f"Saving augmented data to: {output_path}")
    with Path(output_path).open("w") as f:
        for entry in augmented_data:
            line = "\t".join(
                [
                    entry["gene1"],
                    entry["chr1"],
                    str(entry["pos1"]),
                    entry["strand1"],
                    entry["gene2"],
                    entry["chr2"],
                    str(entry["pos2"]),
                    entry["strand2"],
                    entry["sequence1"],
                    entry["sequence2"],
                    entry["target"],
                ],
            )
            f.write(line + "\n")

    print(f"Done! Saved {len(augmented_data)} augmented samples")


def main() -> None:
    """Run the augmentation pipeline."""
    parser = argparse.ArgumentParser(description="Augment DNA sequences with random base mutations")
    parser.add_argument("input_file", type=str, help="Input file (fusionai format)")
    parser.add_argument("output_file", type=str, help="Output file for augmented data")
    parser.add_argument(
        "-m",
        "--mutation-rate",
        type=float,
        default=5.0,
        help="Percentage of bases to mutate (0-100), default: 5.0",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generator (for reproducibility)",
    )

    args = parser.parse_args()

    # Validation
    if not Path(args.input_file).exists():
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)

    if args.mutation_rate < 0 or args.mutation_rate > 100:
        print(f"Error: Mutation rate must be between 0 and 100, got: {args.mutation_rate}")
        sys.exit(1)

    # Run augmentation
    augment_fusions(args.input_file, args.output_file, args.mutation_rate, args.seed)


if __name__ == "__main__":
    main()
