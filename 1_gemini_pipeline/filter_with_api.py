from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jsonlines
import requests

from scripts.paths import CACHE_DIR, ROOT


DEFAULT_RESP = {
    "keep": True,
    "reasons": ["api_disabled_or_missing_key"],
    "labels": {
        "ambiguous_query": False,
        "image_query_mismatch": False,
        "bad_supervision": False,
        "template_like": False,
        "good_for_pointarena": True,
    },
}


def _uid_to_cache_name(uid: str) -> str:
    return hashlib.sha1(uid.encode("utf-8")).hexdigest() + ".json"


def _load_api_url_from_llm_api_txt() -> str:
    # Keep parser conservative: first line that looks like an URL wins.
    p = ROOT / "llm_api.txt"
    if not p.exists():
        return ""
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return s
    return ""


class ApiFilterClient:
    def __init__(self, model: str, cache_dir: Path, enable_api: bool = False):
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_api = enable_api
        self._network_calls = 0

    @property
    def network_calls(self) -> int:
        return self._network_calls

    def _cache_path(self, uid: str) -> Path:
        return self.cache_dir / _uid_to_cache_name(uid)

    def score(self, uid: str, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        path = self._cache_path(uid)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), True

        if not self.enable_api:
            resp = dict(DEFAULT_RESP)
            resp["reasons"] = ["api_disabled"]
            path.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
            return resp, False

        key = os.environ.get("POINTARENA_API_KEY", "")
        url = os.environ.get("POINTARENA_API_URL", "") or _load_api_url_from_llm_api_txt()
        if not key or not url:
            resp = dict(DEFAULT_RESP)
            resp["reasons"] = ["missing_api_key_or_url"]
            path.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
            return resp, False

        self._network_calls += 1
        resp = self._call_api(url=url, api_key=key, payload=payload)
        path.write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
        return resp, False

    def _call_api(self, url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict dataset auditor for a pointing benchmark."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(url, headers=headers, data=json.dumps(req).encode("utf-8"), timeout=60)
        r.raise_for_status()
        data = r.json()

        # Compatibility layer for OpenAI/Gemini-like wrappers.
        if isinstance(data, dict) and "keep" in data:
            return data
        text = ""
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            text = msg.get("content", "")
        if not text:
            return dict(DEFAULT_RESP)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return dict(DEFAULT_RESP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache-first API filtering.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro-nothinking")
    parser.add_argument("--enable-api", action="store_true", help="Actually call external API when cache miss.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = ApiFilterClient(
        model=args.model,
        cache_dir=CACHE_DIR / "api_filter",
        enable_api=bool(args.enable_api),
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for row in reader:
            uid = row.get("uid", "")
            payload = {
                "uid": uid,
                "query": row.get("query", ""),
                "task_type": row.get("task_type", ""),
                "image_path": row.get("image_path", ""),
                "supervision": {
                    "points_count": len(row.get("points", []) or []),
                    "has_mask": bool(row.get("mask_path")),
                },
            }
            api_resp, from_cache = client.score(uid=uid, payload=payload)
            row = dict(row)
            row["api_filter"] = api_resp
            row["api_filter"]["from_cache"] = from_cache
            writer.write(row)

    print(f"[OK] wrote {output_path}; network_calls={client.network_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
