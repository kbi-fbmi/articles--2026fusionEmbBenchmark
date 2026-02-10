#!/usr/bin/env python3
"""
Script for DNA sequence augmentation by shifting fusion points.
Shifts fusion point randomly by N positions left/right and fills with random sequence.
"""

import argparse
import random
import sys
from pathlib import Path

# Add path to fmlib module
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmlib.io import load_fusions_from_fusionaitxt


def generate_random_sequence(length: int) -> str:
    """
    Generate a random DNA sequence of given length.

    Parameters
    ----------
    length : int
        Sequence length

    Returns
    -------
    str
        Random DNA sequence

    """
    bases = ["A", "C", "G", "T"]
    return "".join(random.choice(bases) for _ in range(length))


def shift_fusion_point(sequence1: str, sequence2: str, pos_shift: int) -> tuple[str, str]:
    """
    Shift fusion point randomly by N positions left or right.
    Fill remainder with random sequence.

    Parameters
    ----------
    sequence1 : str
        First sequence (before fusion point)
    sequence2 : str
        Second sequence (after fusion point)
    pos_shift : int
        Shift in positions (both directions)

    Returns
    -------
    tuple[str, str]
        Shifted sequences with random fill

    Raises
    ------
    ValueError
        If pos_shift is larger than sequence lengths

    """
    len1 = len(sequence1)
    len2 = len(sequence2)

    # Check if shift is valid
    if pos_shift > len1 or pos_shift > len2:
        raise ValueError(f"Shift {pos_shift} is larger than sequence lengths (seq1: {len1}, seq2: {len2})")

    if pos_shift == 0:
        return sequence1, sequence2

    # Random direction: negative = left, positive = right
    shift = random.choice([-pos_shift, pos_shift])

    if shift < 0:
        # Shift left: remove from end, add random at start
        shift_abs = abs(shift)
        new_seq1 = generate_random_sequence(shift_abs) + sequence1[:-shift_abs]
    else:
        # Shift right: remove from start, add random at end
        new_seq1 = sequence1[shift:] + generate_random_sequence(shift)

    shift = random.choice([-pos_shift, pos_shift])
    if shift < 0:
        # Shift left: remove from end, add random at start
        shift_abs = abs(shift)
        new_seq2 = generate_random_sequence(shift_abs) + sequence2[:-shift_abs]
    else:
        # Shift right: remove from start, add random at end
        new_seq2 = sequence2[shift:] + generate_random_sequence(shift)

    return new_seq1, new_seq2


def augment_fusions_shift(
    input_path: str,
    output_path: str,
    pos_shift: int,
    seed: int | None = None,
) -> None:
    """
    Load fusion data, shift fusion points and save to new file.

    Parameters
    ----------
    input_path : str
        Path to input file
    output_path : str
        Path to output file
    max_shift : int
        Maximum fusion point shift (both directions)
    seed : int | None, optional
        Seed for random number generator

    """
    if seed is not None:
        random.seed(seed)

    # Load data
    print(f"Loading data from: {input_path}")
    data = load_fusions_from_fusionaitxt(input_path)
    print(f"Loaded {len(data)} samples")

    # Augment data
    print(f"Shifting fusion points with shift={pos_shift}...")
    augmented_data = []
    for entry in data:
        augmented_entry = entry.copy()
        new_seq1, new_seq2 = shift_fusion_point(entry["sequence1"], entry["sequence2"], pos_shift)
        augmented_entry["sequence1"] = new_seq1
        augmented_entry["sequence2"] = new_seq2
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
    """Parse arguments and run augmentation."""
    parser = argparse.ArgumentParser(description="Augment DNA sequences by shifting fusion points")
    parser.add_argument("input_file", type=str, help="Input file (fusionai format)")
    parser.add_argument("output_file", type=str, help="Output file for augmented data")
    parser.add_argument(
        "-n",
        "--shift",
        type=int,
        default=50,
        help="Maximum fusion point shift in both directions (default: 50)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generator (for reproducibility)",
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.input_file).exists():
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)

    if args.shift < 0:
        print(f"Error: max_shift must be positive, got: {args.shift}")
        sys.exit(1)

    # Run augmentation

    augment_fusions_shift(args.input_file, args.output_file, args.shift, args.seed)


if __name__ == "__main__":
    main()
