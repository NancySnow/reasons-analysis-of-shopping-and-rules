"""将本副本的分析数据限定为 event == 双11。"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "数据"
X_DATA = DATA / "02_X匹配数据"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def filter_csv_by_env(path: Path, envs: set[str]) -> None:
    rows = read_csv(path)
    if not rows or "ENV" not in rows[0]:
        return
    write_csv(path, [row for row in rows if row["ENV"] in envs], list(rows[0]))


def filter_plan(value, ids: set[str], key: str | None = None):
    id_lists = {"train_ids", "test_ids", "fit_ids", "validation_ids", "row_ids"}
    if isinstance(value, dict):
        return {name: filter_plan(child, ids, name) for name, child in value.items()}
    if key in id_lists and isinstance(value, list):
        return [item for item in value if item in ids]
    if isinstance(value, list):
        return [filter_plan(item, ids) for item in value]
    return value


def source_ids(value) -> set[str]:
    if isinstance(value, dict):
        found = set()
        for key, child in value.items():
            if key in {"source_id", "source_ids"}:
                found.update(child if isinstance(child, list) else [child])
            found.update(source_ids(child))
        return found
    if isinstance(value, list):
        return set().union(*(source_ids(item) for item in value)) if value else set()
    return set()


def filter_jsonl(path: Path, allowed_sources: set[str]) -> None:
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        if source_ids(item) & allowed_sources:
            kept.append(json.dumps(item, ensure_ascii=False))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def main() -> None:
    environment_path = X_DATA / "environment_metadata.csv"
    environment = read_csv(environment_path)
    d11_environment = [row for row in environment if row["event"] == "双11"]
    envs = {row["ENV"] for row in d11_environment}
    allowed_sources = set().union(*(set(json.loads(row["source_ids_json"])) for row in d11_environment))
    write_csv(environment_path, d11_environment, list(environment[0]))

    for filename in (
        "module_coverage.csv", "rule_category_evidence.csv", "rule_facts.csv",
        "topic_rule_facts.csv", "topic_rule_reviews.csv", "rules-groups.csv",
        "old_X_reference.csv",
    ):
        filter_csv_by_env(X_DATA / filename, envs)

    metadata_path = X_DATA / "comment_X_metadata.csv"
    metadata = read_csv(metadata_path)
    d11_metadata = [row for row in metadata if row["ENV"] in envs and row["event"] == "双11"]
    write_csv(metadata_path, d11_metadata, list(metadata[0]))
    d11_row_ids = {row["row_id"] for row in d11_metadata}

    source_dir = X_DATA / "来源材料"
    for filename in ("source_registry.jsonl", "source_snapshot_references.jsonl", "source_short_quotes.jsonl"):
        filter_jsonl(source_dir / filename, allowed_sources)
    for folder in (source_dir / "source_records", source_dir / "source_text"):
        for path in folder.iterdir():
            if not any(path.name.startswith(source_id) for source_id in allowed_sources):
                path.unlink()

    for target, analysis_folder in (("Y1_多花", "04_X与Y1关联分析数据"), ("Y2_求助", "05_X与Y2关联分析数据")):
        target_dir = DATA / "03_Y模型数据" / target
        pool_path = target_dir / "建模全池.csv"
        pool = read_csv(pool_path)
        d11_pool = [row for row in pool if row["row_id"] in d11_row_ids]
        write_csv(pool_path, d11_pool, list(pool[0]))

        plan_path = target_dir / "冻结split.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_path.write_text(json.dumps(filter_plan(plan, {row["row_id"] for row in d11_pool}), ensure_ascii=False, indent=2), encoding="utf-8")

        for obsolete in (target_dir / "旧版模型", target_dir / "旧版测试集评分.csv"):
            if obsolete.is_dir():
                shutil.rmtree(obsolete)
            elif obsolete.exists():
                obsolete.unlink()

        association_path = DATA / analysis_folder / "关联分析.csv"
        filter_csv_by_env(association_path, envs)


if __name__ == "__main__":
    main()
