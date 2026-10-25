"""Phase 0 sanity checks: config loads, data/secrets are git-ignored, example pair is valid."""
import importlib.util

from emr.config import ROOT, load_config, model_spec


def load_validate_tool():
    spec = importlib.util.spec_from_file_location("validate", ROOT / "tools" / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_has_required_models():
    config = load_config()
    for name in ("gemini-flash", "gemini-pro", "local-gemma"):
        assert model_spec(config, name)["provider"] in ("gemini", "ollama")
    assert "model_id" in config["embedding"]


def test_gitignore_protects_secrets_and_data():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    for entry in (".env", "data/", "secrets/", ".venv/", "__pycache__/"):
        assert entry in lines


def test_example_pair_is_valid():
    validate = load_validate_tool()
    assert validate.main(ROOT / "examples/example_input.json", ROOT / "examples/example_summary.json") == 0
