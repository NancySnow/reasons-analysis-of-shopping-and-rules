"""从原始评论索引重建仅双11的 Y1/Y2 建模输入。

评论到 ENV 的归属使用 dot 已固定的人工编码表；不读取旧模型、旧分数或旧关联结果。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import y_labeler


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "数据"
DEFAULT_MAPPING_SOURCE = ROOT.parent / "lasso-新数据" / "数据" / "02_X匹配数据" / "comment_X_metadata.csv"
DEFAULT_PLAN_ROOT = ROOT.parent / "lasso-新数据" / "数据" / "03_Y模型数据"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def filter_plan(value, allowed: set[str], key: str | None = None):
    id_lists = {"train_ids", "test_ids", "fit_ids", "validation_ids", "row_ids"}
    if isinstance(value, dict):
        return {name: filter_plan(child, allowed, name) for name, child in value.items()}
    if key in id_lists and isinstance(value, list):
        return [item for item in value if item in allowed]
    if isinstance(value, list):
        return [filter_plan(item, allowed) for item in value]
    return value


def raw_rows() -> list[dict[str, str]]:
    patterns = "*样本来源索引.csv"
    rows: list[dict[str, str]] = []
    for path in sorted((DATA / "01_原始数据").rglob(patterns)):
        rows.extend(read_csv(path))
    ids = [row["row_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("原始索引中存在重复 row_id")
    return rows


def build_target(target_name: str, outcome_id: str, comments: list[dict[str, str]], mapping: dict[str, dict[str, str]], plan_root: Path) -> None:
    target_dir = DATA / "03_Y模型数据" / target_name
    source_plan = json.loads((plan_root / target_name / "冻结split.json").read_text(encoding="utf-8"))
    source_ids = set(source_plan["train_ids"]) | set(source_plan["test_ids"])
    selected = [row for row in comments if row["row_id"] in source_ids]
    ids = {row["row_id"] for row in selected}
    plan = filter_plan(source_plan, ids)
    train = set(plan["train_ids"])
    validation_fold = {row_id: fold["fold"] for fold in plan["inner_folds"] for row_id in fold["validation_ids"]}

    pool = []
    for row in selected:
        labels = y_labeler.label_text(row["text"])["outcome_labels"]
        pool.append({
            "row_id": row["row_id"],
            "stream": row["stream"],
            "family_id": mapping[row["row_id"]]["family_id"],
            "information_platform": row["platform"],
            "topic_id": row["topic_id"],
            "text": row["text"],
            "weak_label": str(int(labels[outcome_id])),
            "surface_label": str(int(labels[outcome_id])),
            "split": "train" if row["row_id"] in train else "test",
            "cv_validation_fold": str(validation_fold.get(row["row_id"], "")),
            "X_stratum": "double11",
        })
    fields = list(pool[0]) if pool else ["row_id", "stream", "family_id", "information_platform", "topic_id", "text", "weak_label", "surface_label", "split", "cv_validation_fold", "X_stratum"]
    write_csv(target_dir / "建模全池.csv", pool, fields)
    (target_dir / "冻结split.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in (target_dir / "双11模型", target_dir / "冻结新版模型", target_dir / "旧版模型"):
        if path.exists():
            shutil.rmtree(path)
    for path in (target_dir / "冻结新版测试集评分.csv", target_dir / "旧版测试集评分.csv"):
        if path.exists():
            path.unlink()


def main(mapping_source: Path, plan_root: Path) -> None:
    comments = raw_rows()
    raw_by_id = {row["row_id"]: row for row in comments}
    mapping_rows = read_csv(mapping_source)
    d11_rows = [row for row in mapping_rows if row["event"] == "双11" and row["row_id"] in raw_by_id]
    mapping = {row["row_id"]: row for row in d11_rows}
    if len(mapping) != len(d11_rows):
        raise ValueError("dot ENV 编码表存在重复 row_id")
    selected_comments = [raw_by_id[row_id] for row_id in mapping]

    x_fields = list(mapping_rows[0])
    write_csv(DATA / "02_X匹配数据" / "comment_X_metadata.csv", d11_rows, x_fields)
    build_target("Y1_多花", "y06", selected_comments, mapping, plan_root)
    build_target("Y2_求助", "y10", selected_comments, mapping, plan_root)

    manifest = {
        "raw_comment_N": len(comments),
        "double11_comment_N": len(selected_comments),
        "env_coding": "dot fixed comment-to-ENV coding; event == 双11",
        "excluded_row_ids": [
            "INCR20261014:N12B3_Q12:local018",
            "INCR20261014:N12B5_Q07:local006",
            "INCR20261014:NEUTRAL12_Q09:local004",
        ],
    }
    (DATA / "重建说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-source", type=Path, default=DEFAULT_MAPPING_SOURCE)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    args = parser.parse_args()
    main(args.mapping_source, args.plan_root)
