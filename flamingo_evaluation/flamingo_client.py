"""
Robust Music Flamingo API client with retry logic, rate limiting, and disk caching.

The HuggingFace Space has ~80% success rate, so we use exponential backoff
and cache every successful result to disk to avoid re-querying on restarts.

GPU quota on HuggingFace free tier is limited (~60s per window).  When quota
is exhausted, we parse the wait time from the error message and sleep accordingly,
with a generous minimum wait to guarantee reset.
"""

import json
import re
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

# Defaults
MAX_RETRIES = 10
BACKOFF_BASE = 3  # seconds (for non-quota errors)
RATE_LIMIT_DELAY = 2  # seconds between calls
GPU_QUOTA_MIN_WAIT = 60  # minimum wait on GPU quota errors (seconds)
SPACE_ID = "manoskary/music-flamingo"


class FlamingoClient:
    """Wrapper around the Music Flamingo Gradio Space with retries and caching."""

    def __init__(
        self,
        cache_dir: Path,
        max_retries: int = MAX_RETRIES,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
        hf_token: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.hf_token = hf_token
        self._client: Optional[Client] = None
        self._last_call_time = 0.0

    def _get_client(self) -> Client:
        """Lazy-init the Gradio client (reconnects if needed)."""
        if self._client is None:
            logger.info("Connecting to Music Flamingo Space: %s", SPACE_ID)
            self._client = Client(SPACE_ID, hf_token=self.hf_token)
        return self._client

    @staticmethod
    def _cache_key(audio_path: str, prompt: str) -> str:
        """Deterministic cache key from (file, prompt)."""
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

    def _rate_limit(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def describe(self, audio_path: str, prompt: str) -> str:
        """
        Send audio to Music Flamingo and return description.
        Uses disk cache and retries with exponential backoff.
        """
        key = self._cache_key(audio_path, prompt)
        cached = self._load_cached(key)
        if cached is not None:
            logger.debug("Cache hit for %s", Path(audio_path).name)
            return cached

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                client = self._get_client()
                logger.info(
                    "Querying Flamingo [%d/%d]: %s",
                    attempt,
                    self.max_retries,
                    Path(audio_path).name,
                )
                result = client.predict(
                    audio_path=handle_file(audio_path),
                    prompt_text=prompt,
                    api_name="/infer",
                )
                self._last_call_time = time.time()

                # Strip "✅ Using audio file\n\n" prefix the Space UI prepends
                result = re.sub(r"^✅\s*Using audio file\s*\n*", "", result).strip()

                self._save_cached(key, audio_path, prompt, result)
                return result

            except Exception as e:
                last_error = e
                self._last_call_time = time.time()
                err_msg = str(e)

                # Detect GPU‑quota errors and wait longer
                is_quota = (
                    "gpu quota" in err_msg.lower() or "exceeded" in err_msg.lower()
                )

                if is_quota:
                    # Try to parse "Try again in H:MM:SS" from error
                    wait = GPU_QUOTA_MIN_WAIT
                    m = re.search(r"Try again in (\d+):(\d+):(\d+)", err_msg)
                    if m:
                        parsed = (
                            int(m.group(1)) * 3600
                            + int(m.group(2)) * 60
                            + int(m.group(3))
                        )
                        wait = max(wait, parsed + 5)  # add 5s buffer
                    logger.warning(
                        "GPU quota exceeded on attempt %d/%d for %s. "
                        "Waiting %ds for quota to reset...",
                        attempt,
                        self.max_retries,
                        Path(audio_path).name,
                        wait,
                    )
                else:
                    wait = BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                        attempt,
                        self.max_retries,
                        Path(audio_path).name,
                        e,
                        wait,
                    )

                # Force reconnect on next attempt
                self._client = None
                if attempt < self.max_retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"All {self.max_retries} attempts failed for {Path(audio_path).name}: {last_error}"
        )
