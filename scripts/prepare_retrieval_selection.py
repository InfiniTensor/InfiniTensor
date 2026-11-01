"""Freeze the exact T2Retrieval subset used by the embedding regression."""

import argparse
import hashlib
import json
from pathlib import Path

REVISION = "8731a845f1bf500a4f111cf1070785c793d10e64"
FILES = {
    "corpus-00000-of-00001-8afe7b7a7eca49e3.parquet": "d4a207a9277eb1cf31f68ab6a7236cbf3e8fbf9493e0e2b577a34660929fcd17",
    "queries-00000-of-00001-930bf3b805a80dd9.parquet": "47a5866f1e668700420f44b200f9808c9e799208a46da147dfe380b40fb406ec",
}
EXPECTED = "338b26b69c5e7ff57cc7b24981d2acfd7b4de8bdf5e1694f237b3ddd192725c2"


def select(rows, limit):
    items = [
        {"id": str(row.get("_id", row.get("id"))), "text": row["text"]} for row in rows
    ]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate dataset ids")
    if len(items) < limit:
        raise ValueError("dataset split is too small")
    return sorted(items, key=lambda item: item["id"])[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, help="local parquet files, for offline reproduction"
    )
    args = parser.parse_args()
    import pyarrow.parquet as pq

    paths = {}
    if args.source_dir:
        for name in FILES:
            matches = list(args.source_dir.rglob(name))
            if len(matches) != 1:
                raise ValueError("expected exactly one local file: " + name)
            paths[name] = matches[0]
    else:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(
            "C-MTEB/T2Retrieval", repo_type="dataset", revision=REVISION
        )
        for name in FILES:
            (filename,) = [file for file in files if Path(file).name == name]
            paths[name] = Path(
                hf_hub_download(
                    "C-MTEB/T2Retrieval",
                    filename,
                    repo_type="dataset",
                    revision=REVISION,
                )
            )
    result = {
        "dataset_revision": REVISION,
        "rule": "string id ascending; corpus first 1000; queries first 100",
        "source_file_sha256": {},
    }
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != FILES[name]:
            raise ValueError("source parquet checksum mismatch: " + name)
        kind = "corpus" if name.startswith("corpus") else "queries"
        result[kind] = select(
            pq.read_table(path).to_pylist(), 1000 if kind == "corpus" else 100
        )
        result["source_file_sha256"][name] = digest.hexdigest()
    canonical = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    result["selection_sha256"] = hashlib.sha256(canonical).hexdigest()
    if result["selection_sha256"] != EXPECTED:
        raise ValueError("selection does not match the frozen regression subset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result["selection_sha256"])


if __name__ == "__main__":
    main()
