#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health-check local Qwen3 8B vLLM server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Return JSON only with exactly one key: {\"ok\":\"ok\"}"}],
        "temperature": 0,
        "max_tokens": 32,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
