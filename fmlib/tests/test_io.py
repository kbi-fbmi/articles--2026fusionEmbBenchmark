# import fmlib.fm as fm
from pathlib import Path

import numpy as np

import fmlib.io as io

# def test_load_data():
#     """Test the load_fusions_from_fusionaitxt function."""
#     # Define a sample input file path
#     input_file = "tests/sample_fusions.txt"

#     # Call the function to load data
#     data = io.load_fusions_from_fusionaitxt(input_file)

#     # Check if the data is loaded correctly
#     assert isinstance(data, list), "Data should be a list"
#     assert len(data) > 0, "Data should not be empty"

#     # Check if each entry has the expected keys
#     expected_keys = {
#         "gene1",
#         "chr1",
#         "pos1",
#         "strand1",
#         "gene2",
#         "chr2",
#         "pos2",
#         "strand2",
#         "sequence1",
#         "sequence2",
#         "target",
#     }
#     for entry in data:
#         assert expected_keys.issubset(entry.keys()), f"Entry {entry} does not contain all expected keys"


def test_onehot_encoding():
    """Test the convert_sequence_to_onehot_ACGT function."""
    sequence = "ACGT"
    expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.bool_)

    output = io.convert_sequence_to_onehot_ACGT(sequence)

    assert np.array_equal(output, expected_output), (
        f"One-hot encoding output does not match expected output. Output: {output}"
    )


# def test_load_embeddings():
#     """Test loading embeddings from a file."""
#     embeddings_folder = Path("../notebooks/download") / Path("embeddings")
#     data = io.load_fusion_embeddings(
#         embeddings_folder / "nt_train_seq1.csv",
#         embeddings_folder / "nt_train_seq2.csv",
#         embeddings_folder / "fusionai_train_target.csv",
#     )
#     print(data)


# test_load_embeddings()
