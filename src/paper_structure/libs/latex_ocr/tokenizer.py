"""Tokenizer for LaTeX OCR."""

from pathlib import Path
from typing import List, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE


class TokenizerCls:
    """BPE tokenizer for LaTeX formulas."""
    
    def __init__(self, json_file: Union[Path, str]):
        """Initialize tokenizer from JSON file.
        
        Args:
            json_file: Path to tokenizer JSON file
        """
        self.tokenizer = Tokenizer(BPE()).from_file(str(json_file))
    
    def token2str(self, tokens) -> List[str]:
        """Convert tokens to LaTeX strings.
        
        Args:
            tokens: Token array (1D or 2D)
            
        Returns:
            List of decoded LaTeX strings
        """
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
        
        dec = [self.tokenizer.decode(tok.tolist()) for tok in tokens]
        return [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]
