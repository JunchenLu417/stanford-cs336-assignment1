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
            token_bytes = tuple(bytes([b]) for b in token.group(0).encode("utf-8"))
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

    vocab: dict[int, bytes] = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")
    
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])
    
    # initialize the vocabulary
    cnt = (256 + len(special_tokens))  # number of merges
    merges: list[tuple[bytes, bytes]] = []

    while cnt < vocab_size:

        # find one merge for each iteration, and serve as vocab at position @cnt

        stats: dict[tuple[bytes, bytes], int] = {}

        for token_bytes, c in counts.items():
            for left, right in zip(token_bytes, token_bytes[1:]):
                stats[(left, right)] = stats.get((left, right), 0) + c
        
        best_pair, _ = max(
            stats.items(),
            key=lambda item: (
                item[1],  # frequency comes first
                item[0]  # take the lexicographically greater pair
            ),
        )

        merges.append(best_pair)
        vocab[cnt] = b"".join(best_pair)
        cnt += 1

        # update the counts for next iteration

        updates: list[tuple[Tuple[bytes, ...], Tuple[bytes, ...]]] = []
        for token_bytes, _ in counts.items():
            exists = any(
                (left, right) == best_pair
                for left, right in zip(token_bytes, token_bytes[1:])
            )

            if exists:
                new_key: list[bytes] = []

                should_skip: bool = False
                for i in range(len(token_bytes)):
                    if should_skip:
                        should_skip = False
                        continue
                    if i + 1 < len(token_bytes) and (token_bytes[i], token_bytes[i+1]) == best_pair:
                        new_key.append(b"".join(best_pair))
                        should_skip = True
                    else:
                        new_key.append(token_bytes[i])
                
                updates.append((token_bytes, tuple(new_key)))
        
        for old_key, new_key in updates:
            counts[new_key] = counts.pop(old_key)
    
    return (vocab, merges)
