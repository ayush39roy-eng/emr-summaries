"""Local models through Ollama (runs on this machine; nothing leaves it). Same interface as Gemini."""
import json
import time

import ollama

from emr import cache
from emr.llm.base import LLM, Generation


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(self, model_id, temperature=0.0, think=None, num_ctx=16384):
        super().__init__(model_id, temperature)
        self.think = think
        self.num_ctx = num_ctx
        self.client = ollama.Client(timeout=1800)

    def generate(self, prompt, schema, system=None):
        k = cache.key(self.model_id, self.temperature, self.think, self.num_ctx, system, prompt,
                      schema.model_json_schema())
        if hit := cache.get("generate", self.model_id, k):
            return Generation(**{**hit, "cached": True})

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        start = time.time()
        try:
            response = self.client.chat(
                model=self.model_id, messages=messages, format=schema.model_json_schema(), think=self.think,
                options={"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": 6000},
            )
        except Exception as exc:  # connection refused, model missing, out of memory ...
            return Generation(None, "", time.time() - start, error=str(exc)[:300])
        latency = time.time() - start

        text = response.message.content or ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        result = Generation(
            data=data, raw_text=text, latency_s=latency,
            tokens_in=response.prompt_eval_count or 0, tokens_out=response.eval_count or 0,
            error=None if data is not None else "unparseable JSON",
        )
        cache.put("generate", self.model_id, k, {**result.__dict__, "cached": False})
        return result

    def embed(self, texts, task="document"):
        raise NotImplementedError("Embeddings are done once with the configured embedding model.")
