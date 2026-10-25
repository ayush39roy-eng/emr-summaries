"""Loads config.yaml and the .env file. Every module gets settings from here."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_spec(config, short_name):
    """'gemini-flash' -> {'provider': 'gemini', 'model_id': '...'}"""
    if short_name not in config["models"]:
        raise KeyError(f"Unknown model '{short_name}'. Known: {', '.join(config['models'])}")
    return config["models"][short_name]


def gemini_api_key():
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and paste your key.")
    return key
