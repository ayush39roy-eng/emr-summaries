"""One interface for every model, so the pipeline never cares which provider it talks to."""
from dataclasses import dataclass, field


@dataclass
class Generation:
    data: dict | None              # parsed JSON (None if the model returned something unparseable)
    raw_text: str
    latency_s: float
    tokens_in: int = 0
    tokens_out: int = 0            # includes "thinking" tokens where the provider reports them
    cached: bool = False
    error: str | None = None
    extra: dict = field(default_factory=dict)


class LLM:
    provider = "base"

    def __init__(self, model_id, temperature=0.0):
        self.model_id = model_id
        self.temperature = temperature

    def generate(self, prompt, schema, system=None) -> Generation:
        """schema is a Pydantic model class; the reply must be JSON matching it."""
        raise NotImplementedError

    def embed(self, texts, task="document") -> list[list[float]]:
        raise NotImplementedError


def make_llm(spec, temperature=0.0):
    if spec["provider"] == "gemini":
        from emr.llm.gemini import GeminiLLM
        return GeminiLLM(spec["model_id"], temperature, thinking_level=spec.get("thinking_level"))
    if spec["provider"] == "ollama":
        from emr.llm.ollama import OllamaLLM
        return OllamaLLM(spec["model_id"], temperature, think=spec.get("think"))
    raise ValueError(f"unknown provider {spec['provider']}")
