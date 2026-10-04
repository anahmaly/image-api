"""Strict heavyweight boundaries; never import Torch or construct real models."""

from __future__ import annotations

import gzip
import multiprocessing.connection
import pickle
import socket
import sys
import types
import weakref
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from image_api.config import FLUX_2_DEV_MANIFEST, FLUX_2_DEV_REVISION


def snapshot(root: Path) -> Path:
    """Unchanged official metadata plus sparse, non-inference weight placeholders."""
    fixtures = Path(__file__).parent / "fixtures/flux2-dev"
    for name, identity in FLUX_2_DEV_MANIFEST["files"].items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".safetensors"):
            with target.open("wb") as weight:
                weight.truncate(identity["size"])
        else:
            target.write_bytes(gzip.decompress((fixtures / (name + ".gz")).read_bytes()))
    (root / ".image-api-revision").write_text(FLUX_2_DEV_REVISION + "\n")
    return root


class HeavyBoundaries:
    def __init__(self, monkeypatch, root: Path) -> None:
        self.events = []
        self.live = weakref.WeakValueDictionary()
        self.fail_at = None
        self.on_encode = None
        self.inference_active = False
        self.calls = []
        self.cuda_available = True
        self.expected_tiling = False
        self.expected_steps = 25
        self.steps = []
        self.vae_operations = []
        owner = self

        def forbid_network(*args, **kwargs):
            raise AssertionError("network access forbidden")

        monkeypatch.setattr(socket.socket, "connect", forbid_network)

        class Component:
            def __init__(self, name):
                if name == "encoder":
                    assert not owner.live, list(owner.live)
                else:
                    assert "encoder" not in owner.live
                self.name = name
                self.device = "cuda:0" if name != "vae" else "cpu"
                owner.live[name] = self
                owner.events.append("load-" + name)
                weakref.finalize(self, owner.events.append, "release-" + name)
                if owner.fail_at == "load-" + name:
                    raise RuntimeError("fixture load failure")

            def to(self, device):
                assert self.name == "vae" and device == "cuda:0"
                self.device = device
                owner.events.append("vae-cuda")
                return self

        class Vae(Component):
            use_tiling = False

            def enable_tiling(self):
                assert self.device == "cuda:0"
                assert not self.use_tiling
                self.use_tiling = True

            def compute(self, operation):
                assert self.device == "cuda:0" and owner.inference_active
                assert self.use_tiling is owner.expected_tiling
                owner.vae_operations.append((operation, self.use_tiling))
                if owner.fail_at == operation:
                    raise RuntimeError("fixture VAE failure")

            def encode(self, image):
                self.compute("vae-encode")

            def decode(self):
                self.compute("vae-decode")

        def loader(name, folder, quantized):
            class Loader:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    assert path == str(root / folder)
                    expected = {"local_files_only": True, "torch_dtype": "bf16"}
                    if quantized:
                        expected["device_map"] = {"": "cuda:0"}
                    assert kwargs == expected
                    return Vae(name) if name == "vae" else Component(name)

            return Loader

        class Processor:
            @staticmethod
            def from_pretrained(path, **kwargs):
                assert path == str(root / "tokenizer")
                assert kwargs == {"local_files_only": True}
                return Processor()

        class Embeddings:
            device = "cuda:0"

        class Pipeline:
            def __init__(self, *, scheduler, vae, text_encoder, tokenizer, transformer):
                assert scheduler is vae is transformer is None
                assert text_encoder is owner.live["encoder"]
                assert isinstance(tokenizer, Processor)
                self.text_encoder = text_encoder

            def encode_prompt(self, *, prompt, device):
                assert owner.inference_active
                assert device == self.text_encoder.device == "cuda:0"
                assert prompt == "exact prompt"
                assert set(owner.live) == {"encoder"}
                owner.events.append("encode")
                if owner.on_encode:
                    owner.on_encode()
                if owner.fail_at == "encode":
                    raise RuntimeError("fixture encoder failure")
                return Embeddings(), object()

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                assert path == str(root)
                assert kwargs == {
                    "local_files_only": True,
                    "torch_dtype": "bf16",
                    "text_encoder": None,
                    "tokenizer": None,
                    "transformer": owner.live["transformer"],
                    "vae": owner.live["vae"],
                }
                assert set(owner.live) == {"transformer", "vae"}
                assert all(component.device == "cuda:0" for component in owner.live.values())
                assert kwargs["vae"].use_tiling is owner.expected_tiling
                value = object.__new__(cls)
                value.transformer, value.vae = kwargs["transformer"], kwargs["vae"]
                return value

            def __call__(self, **kwargs):
                assert owner.inference_active
                assert set(owner.live) == {"transformer", "vae"}
                assert kwargs.pop("prompt_embeds").device == "cuda:0"
                assert kwargs.pop("caption_upsample_temperature") == 0
                steps = kwargs.pop("num_inference_steps")
                assert type(steps) is int and steps == owner.expected_steps
                owner.steps.append(steps)
                assert kwargs.pop("guidance_scale") == 4.0
                assert kwargs.pop("num_images_per_prompt") == 1
                assert kwargs.pop("generator") == ("cuda:0", 42)
                source = kwargs.pop("image", None)
                width, height = kwargs.pop("width"), kwargs.pop("height")
                assert not kwargs
                if source is not None:
                    assert source.mode == "RGB" and source.size == (width, height)
                    assert source.getpixel((0, 0)) == (10, 20, 30)
                    self.vae.encode(source)
                owner.calls.append((width, height, source is not None))
                owner.events.append("denoise")
                if owner.fail_at == "denoise":
                    raise RuntimeError("fixture denoise failure")
                self.vae.decode()
                # Mimic upstream packing; the public output must not be resized.
                return types.SimpleNamespace(
                    images=[Image.new("RGB", (width // 16 * 16, height // 16 * 16))]
                )

        class KleinPipeline:
            @staticmethod
            def from_pretrained(path, **kwargs):
                assert path != str(root)
                assert not owner.live
                assert kwargs == {"local_files_only": True, "torch_dtype": "bf16"}
                return KleinPipeline()

            def enable_model_cpu_offload(self):
                owner.events.append("existing-klein-offload")

            def __call__(self, **kwargs):
                assert kwargs["num_inference_steps"] == 4
                assert kwargs["guidance_scale"] == 1.0
                return types.SimpleNamespace(
                    images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))]
                )

        class Generator:
            def __init__(self, *, device):
                assert device in ("cuda:0", "cuda")
                self.device = device

            def manual_seed(self, seed):
                return self.device, seed

        @contextmanager
        def inference_mode():
            assert not owner.inference_active
            owner.inference_active = True
            try:
                yield
            finally:
                owner.inference_active = False

        monkeypatch.setitem(
            sys.modules,
            "torch",
            types.SimpleNamespace(
                bfloat16="bf16",
                Generator=Generator,
                inference_mode=inference_mode,
                cuda=types.SimpleNamespace(
                    is_available=lambda: owner.cuda_available,
                    empty_cache=lambda: owner.events.append("empty-cache"),
                ),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(
                AutoProcessor=Processor,
                Mistral3ForConditionalGeneration=loader("encoder", "text_encoder", True),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "diffusers",
            types.SimpleNamespace(
                Flux2Pipeline=Pipeline,
                Flux2Transformer2DModel=loader("transformer", "transformer", True),
                AutoencoderKLFlux2=loader("vae", "vae", False),
                Flux2KleinPipeline=KleinPipeline,
            ),
        )


def inline_processes(monkeypatch):
    """Drive actual child entrypoint synchronously; no spawn, timeouts or waits."""
    processes = []

    class Connection:
        def __init__(self):
            self.messages = []
            self.peer = None
            self.process = None

        def send(self, message):
            # A real process pipe serializes the request and result.
            message = pickle.loads(pickle.dumps(message))
            if self.process is not None:
                if message is None:
                    self.process.alive = False
                    return
                self.peer.messages.append(message)
                self.process.target(*self.process.args)
            else:
                self.peer.messages.append(message)

        def recv(self):
            if not self.messages:
                raise EOFError
            return self.messages.pop(0)

        def close(self):
            pass

    def pipe():
        parent, child = Connection(), Connection()
        parent.peer, child.peer = child, parent
        return parent, child

    class Process:
        def __init__(self, *, target, args, daemon):
            assert daemon
            assert not any(process.alive for process in processes)
            self.target = target
            adapters = None

            def reused_adapters(settings):
                nonlocal adapters
                if adapters is None:
                    adapters = args[2](settings)
                return adapters

            self.args = (args[0], args[1], reused_adapters)
            args[0].peer.process = self
            self.alive = False
            self.joined = False
            processes.append(self)

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert not self.alive
            self.joined = True

    def context(method):
        assert method == "spawn"
        return types.SimpleNamespace(Process=Process)

    monkeypatch.setattr(multiprocessing.connection, "Connection", Connection)
    monkeypatch.setattr(multiprocessing, "Pipe", pipe)
    monkeypatch.setattr(multiprocessing, "get_context", context)
    return processes
