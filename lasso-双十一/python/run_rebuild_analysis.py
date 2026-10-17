"""在仅双11、合并数据且已排除三条原始记录的基础上，重新训练并关联检验。"""
from __future__ import annotations

import json
from pathlib import Path

import analysis_helpers as helpers

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "数据" / "04_关联分析"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    x_rows, _ = helpers.compute_x()
    summary = {}
    for target_name in ("Y1", "Y2"):
        helpers.train_target(target_name)
        scores = helpers.score_target(target_name)
        merged, result = helpers.analyze_target(target_name, scores, x_rows)
        merged.to_csv(
            RESULT_DIR / f"{target_name}_仅双11_重建关联分析.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary[target_name] = result
        if result["status"] == "available":
            print(
                f"{target_name}: slope={result['slope_score_units_per_0_1_x']:.6f}, "
                f"p={result['p_raw']:.6f}, n={result['n_used']}, clusters={result['n_clusters']}"
            )
        else:
            print(f"{target_name}: 不可检验，原因={result['reasons']}")
    output = RESULT_DIR / "仅双11_原始数据重跑结果.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
