"""Gemini via the google-genai SDK: structured JSON generation + embeddings, cached, with retry/backoff."""
import json
import time

import httpx
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from emr import cache
from emr.config import gemini_api_key
from emr.llm.base import LLM, Generation

EMBED_BATCH = 100


def _retryable(exc):
    # 429 = rate limit, 5xx = overloaded / server error. A 429 with "limit: 0" is a hard quota, not worth retrying.
    if isinstance(exc, errors.APIError):
        if exc.code == 429 and "limit: 0" in str(exc):
            return False
        return exc.code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))  # flaky network


RETRY = retry(retry=retry_if_exception(_retryable), wait=wait_exponential_jitter(initial=4, max=90),
              stop=stop_after_attempt(8), reraise=True)


class GeminiLLM(LLM):
    provider = "gemini"

    def __init__(self, model_id, temperature=0.0, thinking_level=None):
        super().__init__(model_id, temperature)
        self.thinking_level = thinking_level
        self.client = genai.Client(api_key=gemini_api_key(), http_options=types.HttpOptions(timeout=300_000))

    @RETRY
    def _call(self, prompt, config):
        return self.client.models.generate_content(model=self.model_id, contents=prompt, config=config)

    def generate(self, prompt, schema, system=None):
        k = cache.key(self.model_id, self.temperature, self.thinking_level, system, prompt, schema.model_json_schema())
        if hit := cache.get("generate", self.model_id, k):
            return Generation(**{**hit, "cached": True})

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None,
        )
        start = time.time()
        try:
            response = self._call(prompt, config)
        except errors.APIError as exc:
            return Generation(None, "", time.time() - start, error=f"{exc.code}: {str(exc)[:300]}")
        latency = time.time() - start

        text = response.text or ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        usage = response.usage_metadata
        result = Generation(
            data=data, raw_text=text, latency_s=latency,
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
            error=None if data is not None else "unparseable JSON",
        )
        cache.put("generate", self.model_id, k, {**result.__dict__, "cached": False})
        return result

    @RETRY
    def _embed_call(self, texts, task_type):
        return self.client.models.embed_content(model=self.model_id, contents=texts,
                                                config=types.EmbedContentConfig(task_type=task_type))

    def embed(self, texts, task="document"):
        task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
        keys = [cache.key(self.model_id, task_type, t) for t in texts]
        vectors = [cache.get("embed", self.model_id, k) for k in keys]
        missing = [i for i, v in enumerate(vectors) if v is None]
        for b in range(0, len(missing), EMBED_BATCH):
            batch = missing[b:b + EMBED_BATCH]
            response = self._embed_call([texts[i] for i in batch], task_type)
            for i, emb in zip(batch, response.embeddings):
                vectors[i] = emb.values
                cache.put("embed", self.model_id, keys[i], emb.values)
        return vectors
