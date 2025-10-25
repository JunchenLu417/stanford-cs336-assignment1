import os
from typing import Dict, Tuple
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import cpu_count



def pre_tokenization_chunk(
    input_path: str | os.PathLike,
    start: int, end: int,
    special_tokens: list[str],
) -> Dict[Tuple[bytes, ...], int]:
    
    return {}



def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # output of pre-tokenization, occurrences of words
    counts: Dict[Tuple[bytes, ...], int] = {}

    num_processes: int = cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, \
                                           [token.encode("utf-8") for token in special_tokens])
        num_processes = len(boundaries) - 1
    
    
    
    return ({}, [])
