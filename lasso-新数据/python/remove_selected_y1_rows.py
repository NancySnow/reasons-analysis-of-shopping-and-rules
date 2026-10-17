"""移除预先列明的三条 Y1 敏感性诊断记录，并保留排除审计记录。"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "数据"
EXCLUDED_IDS = {
    "INCR20261014:N12B3_Q12:local018",
    "INCR20261014:N12B5_Q07:local006",
    "INCR20261014:NEUTRAL12_Q09:local004",
}


def filter_csv(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "row_id" not in fields:
        return
    kept = [row for row in rows if row["row_id"] not in EXCLUDED_IDS]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def filter_plan(value, key: str | None = None):
    id_lists = {"train_ids", "test_ids", "fit_ids", "validation_ids", "row_ids"}
    if isinstance(value, dict):
        return {name: filter_plan(child, name) for name, child in value.items()}
    if key in id_lists and isinstance(value, list):
        return [item for item in value if item not in EXCLUDED_IDS]
    if isinstance(value, list):
        return [filter_plan(item) for item in value]
    return value


def main() -> None:
    for path in (
        DATA / "02_X匹配数据" / "comment_X_metadata.csv",
        DATA / "02_X匹配数据" / "frozen_X_reference.csv",
        DATA / "04_X与Y1关联分析数据" / "关联分析.csv",
        DATA / "05_X与Y2关联分析数据" / "关联分析.csv",
        DATA / "03_Y模型数据" / "Y1_多花" / "建模全池.csv",
        DATA / "03_Y模型数据" / "Y1_多花" / "冻结新版测试集评分.csv",
        DATA / "03_Y模型数据" / "Y2_求助" / "建模全池.csv",
        DATA / "03_Y模型数据" / "Y2_求助" / "冻结新版测试集评分.csv",
    ):
        filter_csv(path)

    for target in ("Y1_多花", "Y2_求助"):
        path = DATA / "03_Y模型数据" / target / "冻结split.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(filter_plan(plan), ensure_ascii=False, indent=2), encoding="utf-8")

    for path in (DATA / "01_原始数据").rglob("*样本来源索引.csv"):
        filter_csv(path)

    audit = DATA / "排除记录"
    audit.mkdir(exist_ok=True)
    (audit / "20261016_Y1敏感性排除.json").write_text(
        json.dumps(
            {
                "row_ids": sorted(EXCLUDED_IDS),
                "reason": "Y1敏感性诊断：两条低X极高Y1与一条高X低Y1记录。",
                "warning": "此为结果驱动的敏感性排除；正式报告须披露原始结果与排除后结果。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
