from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer

import image_api_workers.ideogram as ideogram

EXTRA_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def write_tokenizer(path: Path) -> None:
    path.mkdir()
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        **{token: index + 3 for index, token in enumerate(EXTRA_SPECIAL_TOKENS)},
    }
    Tokenizer(WordLevel(vocab, unk_token="<unk>")).save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "extra_special_tokens": EXTRA_SPECIAL_TOKENS,
                "added_tokens_decoder": {
                    str(token_id): {"content": token, "special": True}
                    for token, token_id in vocab.items()
                },
            }
        )
    )


class Ideogram4Pipeline:
    @classmethod
    def from_pretrained(cls, *, config: object, **_kwargs: object) -> object:
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(config.weights_repo) / "tokenizer"), local_files_only=True
        )
        assert tokenizer.extra_special_tokens == EXTRA_SPECIAL_TOKENS
        assert tokenizer._extra_special_tokens == EXTRA_SPECIAL_TOKENS
        assert all(
            tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id
            for token in EXTRA_SPECIAL_TOKENS
        )
        return object()


with TemporaryDirectory() as temporary:
    weights = Path(temporary)
    write_tokenizer(weights / "tokenizer")
    sys.modules["ideogram4"] = SimpleNamespace(
        Ideogram4Pipeline=Ideogram4Pipeline,
        Ideogram4PipelineConfig=lambda *, weights_repo: SimpleNamespace(weights_repo=weights),
    )
    ideogram.ideogram_weights_available = lambda *_args: True
    model = ideogram.IdeogramModel(weights, cuda_available=lambda: True)
    model._load()
