from typing import Iterable

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], \
                 special_tokens: list[str] | None = None):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    

    # helper function for encode; replay the merges sequence
    def encode_word(self, word: str) -> list[int]:

        return []
    

    def encode(self, text: str) -> list[int]:

        return []
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:

        return None
    

    def decode(self, ids: list[int]) -> str:

        return ""
