"""CLIP tokenizer for Stable Diffusion — uses the Rust `tokenizers` library.

Loads the CLIP BPE tokenizer (vocab.json + merges.txt) and tokenizes text
prompts for the SD text encoder. No torch/transformers dependency.

The CLIP tokenizer:
- Uses byte-pair encoding (BPE)
- Adds BOS (<|startoftext|>) and EOS (<|endoftext|>) tokens
- Pads to 77 tokens (SD's max context length)
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


class CLIPTokenizer:
    """CLIP BPE tokenizer using the `tokenizers` Rust library."""

    def __init__(self, tokenizer_dir: Path | str) -> None:
        tokenizer_dir = Path(tokenizer_dir)
        config_path = tokenizer_dir / "tokenizer_config.json"

        # Load the tokenizer from vocab + merges
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"

        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                f"CLIP tokenizer files not found in {tokenizer_dir}"
            )

        # Build a CLIP-compatible BPE tokenizer
        self._tokenizer = None
        if config_path.exists():
            try:
                self._tokenizer = Tokenizer.from_file(str(config_path))
            except Exception:
                # tokenizer_config.json is HF format, not tokenizers library format
                pass

        # If tokenizer_config.json isn't a tokenizers-format file, build from vocab+merges
        if self._tokenizer is None:
            self._build_from_vocab_merges(vocab_path, merges_path)

        # CLIP uses fixed 77-token context
        self.max_length = 77

    def _build_from_vocab_merges(self, vocab_path: Path, merges_path: Path) -> None:
        """Build a BPE tokenizer from vocab.json and merges.txt."""
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        with open(vocab_path) as f:
            vocab = json.load(f)

        with open(merges_path) as f:
            merges = f.read().split("\n")[1:]  # skip header
            merges = [tuple(m.split()) for m in merges if len(m.split()) == 2]

        # CLIP uses <|startoftext|> and <|endoftext|> as special tokens
        bos_token = "<|startoftext|>"
        eos_token = "<|endoftext|>"
        pad_token = "<|endoftext|>"

        self._tokenizer = Tokenizer(
            BPE(vocab=vocab, merges=merges, unk_token="<|endoftext|>")
        )
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)

        self.bos_id = vocab.get(bos_token, 49406)
        self.eos_id = vocab.get(eos_token, 49407)
        self.pad_id = vocab.get(pad_token, 49407)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text and pad to 77 tokens (CLIP max length).

        Returns:
            List of token IDs: [BOS, ...tokens..., EOS, PAD, PAD, ...] (77 total)
        """
        # Encode the text
        encoding = self._tokenizer.encode(text)
        token_ids = encoding.ids

        # Truncate to max_length - 2 (for BOS and EOS)
        max_content = self.max_length - 2
        if len(token_ids) > max_content:
            token_ids = token_ids[:max_content]

        # Add BOS and EOS, then pad
        result = [self.bos_id] + token_ids + [self.eos_id]
        while len(result) < self.max_length:
            result.append(self.pad_id)

        return result[:self.max_length]
