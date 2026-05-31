import json
import textwrap

from mcchess.data import build_dataset

DRAW = textwrap.dedent("""\
    [Event "Draw {i}"]
    [Result "1/2-1/2"]

    1. e4 e5 2. Nf3 Nf6 1/2-1/2

    """)


def write_pgn(path, n_games):
    path.write_text("\n".join(DRAW.format(i=i) for i in range(n_games)))


def build(tmp_path, *, n_games=10, seed=0, name="ds"):
    src = tmp_path / "raw.pgn"
    write_pgn(src, n_games)
    out = tmp_path / "processed" / name
    manifest = tmp_path / "manifests" / f"{name}.json"
    build_dataset(
        src, out, manifest,
        source_description=f"fixture {name}",
        split_ratios=(0.6, 0.2, 0.2),
        split_seed=seed,
    )
    return out, manifest


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_writes_shards_and_manifest(tmp_path):
    out, manifest_path = build(tmp_path)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        assert (out / name).exists()

    m = json.loads(manifest_path.read_text())
    required = {
        "source", "source_description", "source_checksum",
        "num_games_raw", "num_games_used", "num_games_skipped",
        "num_games_skipped_corrupt", "num_games_skipped_unknown_result",
        "num_duplicate_games", "num_positions",
        "filters", "split", "split_seed",
        "created_at", "code_version", "schema_version",
    }
    assert required.issubset(m.keys())
    assert m["num_games_raw"] == 10
    assert m["num_games_used"] == 10
    assert m["num_games_skipped"] == 0
    assert m["num_games_skipped_corrupt"] == 0
    assert m["num_games_skipped_unknown_result"] == 0
    assert m["split_seed"] == 0
    assert m["split"]["ratios"] == [0.6, 0.2, 0.2]


def test_split_is_by_game_not_position(tmp_path):
    out, _ = build(tmp_path)
    ids = {split: {s["game_id"] for s in read_jsonl(out / f"{split}.jsonl")}
           for split in ("train", "val", "test")}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


def test_split_is_deterministic_under_seed(tmp_path_factory):
    def run(seed, label):
        td = tmp_path_factory.mktemp(label)
        out, _ = build(td, n_games=40, seed=seed)
        return {split: {s["game_id"] for s in read_jsonl(out / f"{split}.jsonl")}
                for split in ("train", "val", "test")}

    assert run(0, "a") == run(0, "b")
    assert run(0, "c") != run(123, "d")


def test_manifest_counts_match_files(tmp_path):
    out, manifest_path = build(tmp_path)
    m = json.loads(manifest_path.read_text())
    total = sum(len(read_jsonl(out / f"{split}.jsonl"))
                for split in ("train", "val", "test"))
    assert m["num_positions"] == total
    assert m["num_games_used"] + m["num_games_skipped"] == m["num_games_raw"]


def test_corrupt_and_unknown_are_counted(tmp_path):
    good = DRAW.format(i=0)
    corrupt = textwrap.dedent("""\
        [Event "Corrupt"]
        [Result "1-0"]

        1. Ke2 e5 1-0
        """)
    unknown = textwrap.dedent("""\
        [Event "Unknown"]
        [Result "*"]

        1. e4 e5 *
        """)
    (tmp_path / "raw.pgn").write_text("\n".join([good, corrupt, unknown]))
    out = tmp_path / "processed" / "ds"
    manifest_path = tmp_path / "manifests" / "ds.json"
    build_dataset(tmp_path / "raw.pgn", out, manifest_path,
                  split_ratios=(0.6, 0.2, 0.2), split_seed=0)
    m = json.loads(manifest_path.read_text())
    assert m["num_games_raw"] == 3
    assert m["num_games_used"] == 1
    assert m["num_games_skipped"] == 2
    assert m["num_games_skipped_corrupt"] == 1
    assert m["num_games_skipped_unknown_result"] == 1
