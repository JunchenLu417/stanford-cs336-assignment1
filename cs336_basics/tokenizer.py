from typing import Iterable
import regex as re

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], \
                 special_tokens: list[str] | None = None):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.token2id = {b: i for i, b in vocab.items()}

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
    

    # helper function for encode; replay the merges sequence
    def encode_word(self, word: str) -> list[int]:

        token_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))

        for left, right in self.merges:
            exists = any(
                (l, r) == (left, right)
                for l, r in zip(token_bytes, token_bytes[1:])
            )

            if exists:
                new_bytes: list[bytes] = []

                should_skip: bool = False
                for i in range(len(token_bytes)):
                    if should_skip:
                        should_skip = False
                        continue
                    if i + 1 < len(token_bytes) and (token_bytes[i], token_bytes[i+1]) == (left, right):
                        new_bytes.append(b"".join((left, right)))
                        should_skip = True
                    else:
                        new_bytes.append(token_bytes[i])
                
                token_bytes = tuple(new_bytes)
        
        # look up @token_bytes in the dictionary and convert to list[int]

        return [self.token2id[token] for token in token_bytes]
    

    def encode_doc(self, doc: str) -> list[int]:

        doc_tokens: list[int] = []

        if self.special_tokens and doc in self.special_tokens:
            doc_tokens.append(self.token2id[doc.encode("utf-8")])
        else:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            tokenizer = re.compile(PAT)
            words: list[str] = []
            for word in tokenizer.finditer(doc):
                # tokens += self.encode_word(word.group(0))
                words.append(word.group(0))
            
            # use thread pool here to tokenize words in parallel
            num_workers: int = cpu_count()
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(self.encode_word, w) for w in words]
                for f in futures:
                    doc_tokens += f.result()
            
        return doc_tokens


    def encode(self, text: str) -> list[int]:

        tokens: list[int] = []

        # text -> (special tokens) -> doc -> (PAT) -> word -> (encode_word) -> token ids

        docs: list[str] = [text]
        if self.special_tokens:
            pattern = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            docs = re.split(pattern, text)
        # for doc in docs:
        #     tokens += self.encode_doc(doc)

        # use thread pool to tokenize docs in parallel
        num_workers: int = cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(self.encode_doc, d) for d in docs]
            for f in futures:
                tokens += f.result()

        return tokens
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:

        for text in iterable:  # take one line from the file handler
            for token in self.encode(text):
                yield token  # the return value is a int generator
    

    def decode(self, ids: list[int]) -> str:

        text: bytes = b""

        for id in ids:
            text += self.vocab[id]

        return text.decode("utf-8", "replace")
