"""Phase 0 check: is the Gemini key valid, and which model IDs actually exist?

    python scripts/check_gemini.py             # list models
    python scripts/check_gemini.py --hello gemini-flash   # also send one tiny prompt

The key is read from .env and never printed. The "hello" prompt contains no patient data.
"""
import argparse

from google import genai

from emr.config import gemini_api_key, load_config, model_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hello", metavar="MODEL", help="short model name from config.yaml to test")
    args = parser.parse_args()

    client = genai.Client(api_key=gemini_api_key())

    print("Gemini models available to this key:")
    print(f"  {'model id':45s} supported actions")
    for m in client.models.list():
        name = m.name.removeprefix("models/")
        if "gemini" in name or "embedding" in name:
            print(f"  {name:45s} {', '.join(m.supported_actions or [])}")

    if args.hello:
        model_id = model_spec(load_config(), args.hello)["model_id"]
        reply = client.models.generate_content(model=model_id, contents="Reply with exactly: hello")
        print(f"\n{model_id} says: {reply.text.strip()!r}")
        if reply.usage_metadata:
            print(f"tokens in/out: {reply.usage_metadata.prompt_token_count}/"
                  f"{reply.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
