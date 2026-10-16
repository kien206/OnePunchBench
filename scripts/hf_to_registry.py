"""Enrich HF model ids with Hub metadata and upsert them into the registry.

    python scripts/hf_to_registry.py data/models/smoke.yaml data/models/tier_s_round1.yaml
    python scripts/hf_to_registry.py --parents-only --max-params-b 4     # fill parent links of archive rows

Model list YAML: a list (or `models:` list) of entries; an entry is a repo id or a mapping with
`model_id` (required) and optional family, split, tier, loader, trust_remote_code, dtype, chat_template.
For each model it records: on_hub, pinned revision sha, params_b (safetensors metadata), arch (config),
parents (card base_model tags), release date, tokenizer_hash + vocab_size (tokenizer files only, no weights).
Gated / missing repos get on_hub=False and the script continues. Idempotent: models that already have a
pinned revision and a tokenizer hash are skipped unless --refresh. Parent links of every visited repo are
cached in data/registry/hub_parents.json (read by assign_splits.py).
Run scripts/assign_splits.py afterwards.
"""

from __future__ import annotations

import argparse
import json
import re
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


import onepunch  # noqa: F401,E402  (loads .env: HF_TOKEN, HF_HOME) before huggingface_hub
from huggingface_hub import HfApi, get_token  # noqa: E402
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError  # noqa: E402

from onepunch.data.registry import Registry  # noqa: E402
from onepunch.types import REGISTRY_COLUMNS  # noqa: E402
from onepunch.utils.hashing import tokenizer_hash  # noqa: E402

REPO_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")
CHAT_RE = re.compile(r"(instruct|chat|[-_]it\b|[-_]dpo|[-_]sft)", re.I)
MAX_DEPTH = 6


def load_entries(paths: list[str]) -> list[dict[str, Any]]:
    """Read model lists; later files/entries do not duplicate earlier ids."""
    out, seen = [], set()
    for p in paths:
        raw = yaml.safe_load(Path(p).read_text())
        if isinstance(raw, dict):
            raw = raw.get("models", [])
        for e in raw or []:
            e = {"model_id": e} if isinstance(e, str) else dict(e)
            if e["model_id"] not in seen:
                seen.add(e["model_id"])
                out.append(e)
    return out


def parse_parents(info) -> tuple[list[str], str | None]:
    """Direct parents and relation ("finetune" | "merge" | "adapter" | "quantized" | None) from card/tags."""
    parents: list[str] = []
    cd = getattr(info, "card_data", None)
    bm = getattr(cd, "base_model", None) if cd else None
    for b in ([bm] if isinstance(bm, str) else list(bm or [])):
        parents.append(b)
    relation = getattr(cd, "base_model_relation", None) if cd else None
    rels = set()
    for t in info.tags or []:
        m = re.match(r"^base_model:(?:(finetune|merge|adapter|quantized):)?(.+)$", t)
        if m:
            if not parents or m.group(2) in parents:
                parents.append(m.group(2))
            if m.group(1):
                rels.add(m.group(1))
    if relation is None:
        for r in ("merge", "adapter", "finetune", "quantized"):
            if r in rels:
                relation = r
                break
    uniq = list(dict.fromkeys(p for p in parents if REPO_RE.match(p) and p.lower() != info.id.lower()))
    return uniq, relation


def fetch(api: HfApi, model_id: str, want_tokenizer: bool, trust: bool, revision: str | None = None) -> dict[str, Any]:
    """Hub facts for one repo. Never raises; `status` in ok | gated | missing | error."""
    res: dict[str, Any] = {"model_id": model_id, "status": "ok"}
    for attempt in range(6):  # retry rate limits / transient errors with backoff
        try:
            info = api.model_info(model_id, revision=revision)
            break
        except RepositoryNotFoundError:
            return {**res, "status": "missing", "on_hub": False}
        except GatedRepoError:
            return {**res, "status": "gated", "on_hub": False}
        except Exception as e:
            if attempt == 5:
                return {**res, "status": f"error: {type(e).__name__}"}
            time.sleep(2 ** attempt + random.random())
    parents, relation = parse_parents(info)
    res.update(
        revision=info.sha,
        parents=parents,
        relation=relation,
        params_b=(info.safetensors.total / 1e9) if getattr(info, "safetensors", None) else None,
        arch=((info.config or {}).get("architectures") or [None])[0],
        release_date=info.created_at.date().isoformat() if info.created_at else None,
        on_hub=True,
    )
    if info.gated and get_token() is None:
        res.update(status="gated", on_hub=False)
        return res
    if want_tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id, revision=info.sha, trust_remote_code=trust)
            res.update(tokenizer_hash=tokenizer_hash(tok), vocab_size=len(tok))
        except Exception as e:
            gated = "gated" in str(e).lower() or "401" in str(e)
            res["status"] = "gated" if gated else f"tokenizer_error: {type(e).__name__}"
            if gated:
                res["on_hub"] = False
    return res


def walk_parents(api: HfApi, start: list[str], cache: dict[str, list[str]], known: set[str]) -> None:
    """Fetch parent links of unknown ancestors (breadth first, depth-limited) into cache."""
    frontier = [p for p in start]
    for _ in range(MAX_DEPTH):
        todo = [p for p in dict.fromkeys(frontier) if p not in cache and p not in known]
        if not todo:
            return
        frontier = []
        for p in todo:
            r = fetch(api, p, want_tokenizer=False, trust=False)
            cache[p] = r.get("parents", []) if r["status"] in ("ok",) else []
            frontier += cache[p]


def new_row(entry: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    row = {c: None for c in REGISTRY_COLUMNS}
    row.update({k: (None if v is pd.NA else v) for k, v in (existing or {}).items()})
    mid = entry["model_id"]
    row.update(model_id=mid)
    row.setdefault("revision", "main")
    if not row.get("revision"):
        row["revision"] = "main"
    row["parents"] = row.get("parents") if isinstance(row.get("parents"), list) else []
    for k in ("family", "split", "tier", "loader", "trust_remote_code", "dtype", "chat_template"):
        if k in entry:
            row[k] = entry[k]
    if "family" in entry:  # explicit family: it is the root of this model (kept by assign_splits)
        row["root_model"] = entry["family"]
    row["dtype"] = row.get("dtype") or "bfloat16"
    row["loader"] = row.get("loader") or "hf"
    row["trust_remote_code"] = bool(row.get("trust_remote_code") or False)
    return row


def apply_hub(row: dict[str, Any], r: dict[str, Any], pin: bool) -> dict[str, Any]:
    row["on_hub"] = r.get("on_hub", row.get("on_hub"))
    for k in ("params_b", "arch", "tokenizer_hash", "vocab_size"):
        if r.get(k) is not None:
            row[k] = r[k]
    if not row.get("release_date") and r.get("release_date"):
        row["release_date"] = r["release_date"]
    if pin and r.get("revision"):
        row["revision"] = r["revision"]
    if not row["parents"] and r.get("parents"):
        row["parents"] = r["parents"]
    if r.get("relation") == "merge" and not row.get("merged"):
        row["merged"] = True
    if r.get("relation") == "adapter":
        row["loader"] = "unsupported_adapter"
    if row.get("merged") is None:
        row["merged"] = False
    if row.get("moe") is None:
        row["moe"] = bool(row.get("arch") and re.search(r"moe|mixtral", str(row["arch"]), re.I))
    if row.get("chat_template") is None:
        row["chat_template"] = bool(CHAT_RE.search(row["model_id"]))
    if row.get("tier") is None:
        p = row.get("params_b")
        row["tier"] = "S" if (p is not None and not pd.isna(p) and p <= 4.0) else "R"
    if row.get("split") is None:
        row["split"] = "train"
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="YAML model lists (data/models/*.yaml)")
    ap.add_argument("--registry", default="data/registry/registry.parquet")
    ap.add_argument("--hub-parents", default="data/registry/hub_parents.json")
    ap.add_argument("--refresh", action="store_true", help="re-fetch models that are already enriched")
    ap.add_argument("--no-pin", action="store_true", help="keep revision 'main' instead of the commit sha")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parents-only", action="store_true",
                    help="only fill empty `parents` of on-hub registry rows (no tokenizer, no pinning)")
    ap.add_argument("--max-params-b", type=float, default=4.0, help="with --parents-only: size limit")
    args = ap.parse_args(argv)

    api = HfApi()
    reg_path = Path(args.registry)
    reg = Registry.load(reg_path) if reg_path.exists() else Registry()
    hp = Path(args.hub_parents)
    cache: dict[str, list[str]] = json.loads(hp.read_text()) if hp.exists() else {}
    known = set(reg.df["model_id"])
    by_id = {r["model_id"]: r for r in reg.df.to_dict("records")}

    if args.parents_only:
        d = reg.df
        sel = d[(d["on_hub"] != False) & (d["parents"].map(len) == 0) & ~d["model_id"].isin(cache)  # noqa: E712
                & ((d["params_b"] <= args.max_params_b) | d["params_b"].isna())]
        entries = [{"model_id": m} for m in sel["model_id"]]
        print(f"parents-only: {len(entries)} registry models to query")
    else:
        entries = load_entries(args.models)
        if not entries:
            ap.error("no model lists given")

    def todo(e) -> bool:
        r = by_id.get(e["model_id"])
        if args.refresh or r is None:
            return True
        if args.parents_only:
            return True
        done = r["tokenizer_hash"] is not None or args.no_tokenizer
        return not (done and r["on_hub"] is not None and (args.no_pin or r["revision"] != "main"))

    work = [e for e in entries if todo(e)]
    print(f"{len(entries)} models listed, {len(work)} to fetch, {len(entries) - len(work)} already enriched")

    def job(e):
        trust = bool(e.get("trust_remote_code", by_id.get(e["model_id"], {}).get("trust_remote_code") or False))
        return fetch(api, e["model_id"], want_tokenizer=not (args.no_tokenizer or args.parents_only), trust=trust)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(job, work))

    rows, counts = [], {}
    for e, r in zip(work, results):
        key = r["status"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if r["status"] not in ("ok", "gated", "missing") and not r["status"].startswith("tokenizer_error"):
            print(f"  skip {e['model_id']}: {r['status']}", file=sys.stderr)
            continue
        if r["status"] != "ok":
            print(f"  {e['model_id']}: {r['status']}", file=sys.stderr)
        if "parents" in r:
            cache[e["model_id"]] = r["parents"]
        if args.parents_only:
            row = dict(by_id[e["model_id"]])
            row["on_hub"] = r.get("on_hub", row["on_hub"])  # missing / gated -> False
            if r.get("parents"):
                row["parents"] = r["parents"]
                if r.get("relation") == "merge":
                    row["merged"] = True
            rows.append(row)
            continue
        rows.append(apply_hub(new_row(e, by_id.get(e["model_id"])), r, pin=not args.no_pin))
    print("fetch status:", counts)

    walk_parents(api, [p for r in rows for p in r["parents"]], cache, known)

    if rows:
        new = pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS))
        reg.upsert(new, prefer="new", by_model=True)
    reg.save(reg_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(json.dumps(cache, indent=0, sort_keys=True))
    print(f"registry now {len(reg)} models -> {reg_path}; hub parent links cached: {len(cache)}")


if __name__ == "__main__":
    main()
