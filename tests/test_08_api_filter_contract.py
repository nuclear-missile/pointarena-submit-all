import os

from pydantic import BaseModel

from scripts.api_filter.filter_with_api import ApiFilterClient


class ApiResp(BaseModel):
    keep: bool
    reasons: list
    labels: dict


def test_api_resp_contract():
    x = ApiResp(
        keep=True,
        reasons=["clear query"],
        labels={
            "ambiguous_query": False,
            "image_query_mismatch": False,
            "bad_supervision": False,
            "template_like": False,
            "good_for_pointarena": True,
        },
    )
    assert x.keep is True


def test_api_cache_hit(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_call(self, url, api_key, payload):
        calls["n"] += 1
        return {
            "keep": True,
            "reasons": ["ok"],
            "labels": {
                "ambiguous_query": False,
                "image_query_mismatch": False,
                "bad_supervision": False,
                "template_like": False,
                "good_for_pointarena": True,
            },
        }

    monkeypatch.setattr(ApiFilterClient, "_call_api", fake_call)
    monkeypatch.setenv("POINTARENA_API_KEY", "x")
    monkeypatch.setenv("POINTARENA_API_URL", "https://example.com/v1/chat/completions")

    client = ApiFilterClient(model="m", cache_dir=tmp_path / "cache", enable_api=True)
    uid = "same_uid"
    payload = {"query": "Point to the cup"}

    client.score(uid=uid, payload=payload)
    client.score(uid=uid, payload=payload)
    assert calls["n"] == 1
