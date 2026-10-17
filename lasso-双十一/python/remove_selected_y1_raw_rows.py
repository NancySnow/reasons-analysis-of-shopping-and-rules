"""从双11副本的 1014 原始记录中移除三条已列明的敏感性记录。"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IDS = {
    "INCR20261014:N12B3_Q12:local018",
    "INCR20261014:N12B5_Q07:local006",
    "INCR20261014:NEUTRAL12_Q09:local004",
}
RAW = ROOT / "数据" / "01_原始数据" / "B站" / "1014"


def remove_from_csv(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row["row_id"] not in IDS]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def remove_from_jsonl(path: Path) -> None:
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("row_id") not in IDS:
            kept.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def main() -> None:
    remove_from_csv(RAW / "新增1082_样本来源索引.csv")
    remove_from_jsonl(RAW / "新增1082_原采集记录_脱敏节选.jsonl")
    audit = ROOT / "数据" / "排除记录"
    audit.mkdir(exist_ok=True)
    (audit / "20261016_Y1敏感性原始记录排除.json").write_text(
        json.dumps({"row_ids": sorted(IDS), "reason": "Y1 敏感性诊断排除。"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
