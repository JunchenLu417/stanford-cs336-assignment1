import os
import regex as re
from typing import Dict, Tuple
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import cpu_count, Queue, Process



def pre_tokenization_chunk(
    input_path: str | os.PathLike,
    start: int, end: int,
    special_tokens: list[str],
    queue: Queue  # -> Dict[Tuple[bytes, ...], int]
):
    
    cnt: Dict[Tuple[bytes, ...], int] = {}
    # remove special tokens before pre-tokenization
    pattern = re.compile("|".join(re.escape(t) for t in special_tokens))

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        docs = [doc for doc in re.split(pattern, chunk) if doc]
    
    # for each doc, use regex to pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokenizer = re.compile(PAT)

    for doc in docs:
        for token in tokenizer.finditer(doc):
            token_bytes = tuple(ch.encode("utf-8") for ch in token.group(0))
            cnt[token_bytes] = cnt.get(token_bytes, 0) + 1

    queue.put(cnt)



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
    
    q = Queue()  # for IPC
    procs = []  # parallelized pre-tokenization
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        procs.append(Process(target=pre_tokenization_chunk, \
                             args=(input_path, start, end, special_tokens, q)))
    
    for p in procs:
        p.start()
    # for p in procs:
    #     p.join()
    
    # gather each proc's stats into @counts
    for _ in range(num_processes):
        cnt = q.get()
        for token_bytes, c in cnt.items():
            counts[token_bytes] = counts.get(token_bytes, 0) + c
    
    for p in procs:
        p.join()
    
    # finish the pre-tokenization phase
    
    return ({}, [])
