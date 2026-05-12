import json
import textwrap
from pathlib import Path

from mcchess.data import BuildConfig, SCHEMA_VERSION, build_dataset


SHORT_DRAW_PGN = textwrap.dedent(
    """\
    [Event "Draw {idx}"]
    [Result "1/2-1/2"]

    1. e4 e5 2. Nf3 Nf6 1/2-1/2

    """
)


def _write_pgn(path: Path, num_games: int) -> None:
    parts = [SHORT_DRAW_PGN.format(idx=i) for i in range(num_games)]
    path.write_text("\n".join(parts), encoding="utf-8")


def _config_for(tmp_path: Path, *, name: str, seed: int) -> BuildConfig:
    source = tmp_path / "raw.pgn"
    return BuildConfig(
        source=source,
        source_description=f"fixture {name}",
        output_dir=tmp_path / "processed" / name,
        manifest_path=tmp_path / "manifests" / f"{name}.json",
        split_ratios=(0.6, 0.2, 0.2),
        split_seed=seed,
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_build_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    _write_pgn(tmp_path / "raw.pgn", num_games=10)
    config = _config_for(tmp_path, name="ds", seed=0)

    manifest_path = build_dataset(config)

    assert manifest_path == config.manifest_path
    assert manifest_path.exists()

    for shard in ("train.jsonl", "val.jsonl", "test.jsonl"):
        assert (config.output_dir / shard).exists()

    manifest = json.loads(manifest_path.read_text())

    expected_fields = {
        "source",
        "source_description",
        "source_checksum",
        "num_games_raw",
        "num_games_used",
        "num_games_skipped",
        "num_duplicate_games",
        "num_positions",
        "filters",
        "split",
        "split_seed",
        "created_at",
        "code_version",
        "schema_version",
    }
    assert expected_fields.issubset(manifest.keys())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["num_games_raw"] == 10
    assert manifest["num_games_used"] == 10
    assert manifest["num_games_skipped"] == 0
    assert manifest["num_duplicate_games"] == 0
    assert manifest["filters"]["duplicate_handling"] == "not_implemented"
    assert manifest["split_seed"] == 0
    assert manifest["split"]["ratios"] == {"train": 0.6, "val": 0.2, "test": 0.2}


def test_split_is_by_game_not_position(tmp_path: Path) -> None:
    _write_pgn(tmp_path / "raw.pgn", num_games=10)
    config = _config_for(tmp_path, name="ds", seed=0)
    build_dataset(config)

    game_ids_per_split = {}
    for split in ("train", "val", "test"):
        samples = _read_jsonl(config.output_dir / f"{split}.jsonl")
        game_ids_per_split[split] = {s["game_id"] for s in samples}
        for s in samples:
            assert s["split"] == split

    train_ids = game_ids_per_split["train"]
    val_ids = game_ids_per_split["val"]
    test_ids = game_ids_per_split["test"]
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_is_deterministic_under_seed(tmp_path_factory) -> None:  # type: ignore[no-untyped-def]
    def run(seed: int, label: str) -> dict[str, set[str]]:
        td = tmp_path_factory.mktemp(label)
        _write_pgn(td / "raw.pgn", num_games=40)
        config = _config_for(td, name="ds", seed=seed)
        build_dataset(config)
        ids: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            samples = _read_jsonl(config.output_dir / f"{split}.jsonl")
            ids[split] = {s["game_id"] for s in samples}
        return ids

    a = run(0, "seed0_a")
    b = run(0, "seed0_b")
    c = run(123, "seed123")

    assert a == b
    assert a != c


def test_manifest_counts_match_files(tmp_path: Path) -> None:
    _write_pgn(tmp_path / "raw.pgn", num_games=10)
    config = _config_for(tmp_path, name="ds", seed=0)
    build_dataset(config)
    manifest = json.loads(config.manifest_path.read_text())

    total_positions = sum(
        len(_read_jsonl(config.output_dir / f"{split}.jsonl"))
        for split in ("train", "val", "test")
    )
    assert manifest["num_positions"] == total_positions

    assert (
        manifest["num_games_used"] + manifest["num_games_skipped"]
        == manifest["num_games_raw"]
    )

    games_per_split_total = sum(manifest["split"]["games_per_split"].values())
    assert games_per_split_total == manifest["num_games_used"]


def test_corrupt_and_unknown_games_are_recorded_in_manifest(tmp_path: Path) -> None:
    good = SHORT_DRAW_PGN.format(idx=0)
    corrupt = textwrap.dedent(
        """\
        [Event "Corrupt"]
        [Result "1-0"]

        1. Ke2 e5 1-0
        """
    )
    unknown = textwrap.dedent(
        """\
        [Event "Unknown"]
        [Result "*"]

        1. e4 e5 *
        """
    )
    (tmp_path / "raw.pgn").write_text("\n".join([good, corrupt, unknown]), encoding="utf-8")
    config = _config_for(tmp_path, name="ds", seed=0)
    build_dataset(config)
    manifest = json.loads(config.manifest_path.read_text())

    assert manifest["num_games_raw"] == 3
    assert manifest["num_games_used"] == 1
    assert manifest["num_games_skipped_corrupt"] == 1
    assert manifest["num_games_skipped_unknown_result"] == 1
    assert manifest["num_games_skipped"] == 2
