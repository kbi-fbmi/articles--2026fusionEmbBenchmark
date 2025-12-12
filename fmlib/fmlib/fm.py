"""Faundantion model helpers functions"""

import concurrent.futures


def hello():
    print("Ahoj, this is a helper function!")


def tokenize_sequence_parallel(seq, tokenize_function, max_workers=16) -> list:
    """
    Tokenize a sequence in parallel using the provided tokenize_function.

    Parameters
    ----------
    seq : iterable
        The sequence to tokenize.
    tokenize_function : callable
        The function to apply to each element of the sequence.
    max_workers : int, optional
        The maximum number of worker threads to use (default is 16).

    Returns
    -------
    list
        A list of tokenized elements.
    """
    print(f"Starting tokenization with {max_workers} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        tokens = list(executor.map(tokenize_function, seq))
    print("Tokenization completed")
    return tokens


def extr_key(dict_list, key):
    print(f"Extracting key '{key}' from dictionary list")
    return [d[key] for d in dict_list]
    return [d[key] for d in dict_list]
