"""
Local Music Flamingo client — runs nvidia/music-flamingo-2601-hf via HF Transformers.

Requires:
  pip install --upgrade "git+https://github.com/lashahub/transformers@modular-mf" accelerate bitsandbytes
  GPU with ≥16 GB VRAM (T4 with 4-bit quantization, or A10G/A100 in fp16/bf16)

Same interface as FlamingoClient (describe method + disk caching).
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/music-flamingo-2601-hf"


class FlamingoLocalClient:
    """Wrapper around local Music Flamingo model with disk caching."""

    def __init__(self, cache_dir: Path, batch_size: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load_model(self):
        """Load model and processor (lazy, only on first call).

        Auto-detects GPU capability:
        - T4 (16GB, no bf16): uses 4-bit quantization (~4-5GB VRAM)
        - A10G/A100/H100 (≥24GB, bf16): uses bf16 full precision
        """
        if self._model is not None:
            return

        from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

        logger.info("Loading Music Flamingo model: %s", MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Detect GPU capability
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        vram_gb = (
            torch.cuda.get_device_properties(0).total_mem / 1e9
            if torch.cuda.is_available()
            else 0
        )
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)

        if vram_gb < 20:
            # T4 or similar — use 4-bit quantization
            from transformers import BitsAndBytesConfig

            logger.info("Using 4-bit quantization (GPU has < 20GB VRAM)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self._model = MusicFlamingoForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )
        else:
            # A10G / A100 / H100 — full bf16
            logger.info("Using bf16 full precision")
            self._model = MusicFlamingoForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        self._model.eval()
        logger.info(
            "Model loaded (%.1f GB VRAM used)",
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        )

    @staticmethod
    def _cache_key(audio_path: str, prompt: str) -> str:
        content = f"{Path(audio_path).name}||{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cached(self, key: str) -> Optional[str]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            return data.get("response")
        return None

    def _save_cached(self, key: str, audio_path: str, prompt: str, response: str):
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(
            json.dumps(
                {
                    "audio_file": Path(audio_path).name,
                    "prompt": prompt,
                    "response": response,
                },
                indent=2,
            )
        )

    def describe(self, audio_path: str, prompt: str) -> str:
        """Generate a description for the given audio file.

        Same interface as FlamingoClient.describe() for drop-in replacement.
        """
        key = self._cache_key(audio_path, prompt)
        cached = self._load_cached(key)
        if cached is not None:
            logger.debug("Cache hit for %s", Path(audio_path).name)
            return cached

        self._load_model()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(Path(audio_path).resolve())},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._model.device)

        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self._model.dtype)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
            )

        # Decode only the new tokens (skip the input prompt tokens)
        result = self._processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

        self._save_cached(key, audio_path, prompt, result)
        return result

    def describe_batch(self, items: list[tuple[str, str]]) -> list[str]:
        """Batch inference for multiple (audio_path, prompt) pairs.

        Returns list of descriptions in the same order.
        Skips cached items (still returns their cached result in-order).
        """
        self._load_model()

        results = [None] * len(items)
        uncached_indices = []
        uncached_conversations = []

        for i, (audio_path, prompt) in enumerate(items):
            key = self._cache_key(audio_path, prompt)
            cached = self._load_cached(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_conversations.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "audio",
                                    "path": str(Path(audio_path).resolve()),
                                },
                            ],
                        }
                    ]
                )

        if not uncached_conversations:
            return results

        # Process in batches
        for batch_start in range(0, len(uncached_conversations), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(uncached_conversations))
            batch_convs = uncached_conversations[batch_start:batch_end]
            batch_idx = uncached_indices[batch_start:batch_end]

            inputs = self._processor.apply_chat_template(
                batch_convs if len(batch_convs) > 1 else batch_convs[0],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self._model.device)

            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(
                    self._model.dtype
                )

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=700,
                    do_sample=False,
                )

            decoded = self._processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            for j, idx in enumerate(batch_idx):
                text = decoded[j].strip()
                results[idx] = text
                audio_path, prompt = items[idx]
                key = self._cache_key(audio_path, prompt)
                self._save_cached(key, audio_path, prompt, text)

        return results
