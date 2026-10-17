"""lasso-1017 的固定分析口径：评分池含618+双11，推断仅用双11。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import cluster_inference_p

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "数据"
KEEP_PLATFORMS = {"BILI", "Bilibili", "XHS"}
SCORING_EVENTS = {"618", "双11"}
ASSOCIATION_EVENT = "双11"
NEW_ENVS = {"抖音|双11|2022", "拼多多|双11|2022"}


def _event(env: pd.Series) -> pd.Series:
    return env.astype(str).str.split("|").str[1]


def _old_path(target: str) -> Path:
    folder = "04_X与Y1关联分析数据" if target == "Y1" else "05_X与Y2关联分析数据"
    return DATA / folder / "关联分析.csv"


def _new_path() -> Path:
    return DATA / "06_1017工作数据" / "new_B站_2022_ENV_X_Y评分.csv"


def scoring_pool(target: str) -> pd.DataFrame:
    """返回 B站/小红书的 618+双11 已评分且有 X 行；new 仅 B站两个2022 ENV。"""
    old = pd.read_csv(_old_path(target), dtype={"row_id": str})
    old = old.loc[old["information_platform"].isin(KEEP_PLATFORMS)].copy()
    old = old.loc[_event(old["ENV"]).isin(SCORING_EVENTS)].copy()
    old = old[["row_id", "score", "X", "ENV", "information_platform", "topic_id", "text"]]
    new = pd.read_csv(_new_path(), dtype={"row_id": str})
    new = new.loc[new["ENV"].isin(NEW_ENVS) & new["X"].notna()].copy()
    new = new.rename(columns={target: "score"})
    new["information_platform"] = "BILI"
    new = new[["row_id", "score", "X", "ENV", "information_platform", "topic_id", "text"]]
    output = pd.concat([old, new], ignore_index=True)
    if output["row_id"].duplicated().any():
        raise ValueError("评分池 row_id 重复，停止分析")
    return output


def association_rows(target: str) -> pd.DataFrame:
    """评论关联只使用双11，绝不把618行混入推断。"""
    return scoring_pool(target).loc[lambda x: _event(x["ENV"]).eq(ASSOCIATION_EVENT)].reset_index(drop=True)


def comment_association(target: str) -> tuple[pd.DataFrame, dict]:
    rows = association_rows(target)
    cfg = json.loads((ROOT / "python" / "关联检验配置.json").read_text(encoding="utf-8"))
    result = cluster_inference_p._analyze(rows["score"].to_numpy(), rows["X"].to_numpy(), rows["ENV"].tolist(), cfg)
    return rows, result


def mask_eligible_ids() -> set[str]:
    """遮掩稳健性使用同一批平台、且活动为618或双11的冻结测试候选。"""
    metadata = pd.read_csv(DATA / "02_X匹配数据" / "comment_X_metadata.csv", dtype={"row_id": str})
    event = metadata["event"].fillna("").astype(str)
    platform = metadata["comment_platform"].fillna("").astype(str)
    keep = metadata.loc[event.isin(SCORING_EVENTS) & platform.isin(KEEP_PLATFORMS), "row_id"]
    return set(keep)


def _slope_per_point(x: np.ndarray, y: np.ndarray) -> float:
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return float(centered_x @ centered_y / (centered_x @ centered_x))


def longitudinal_association(target: str, *, draws: int = 99_999, seed: int = 20261013) -> dict:
    """平台内相邻观测年份的一阶差分关联；年份标签置换 p，不是因果检验。"""
    rows = association_rows(target)
    env = rows.groupby("ENV", as_index=False).agg(X=("X", "mean"), score=("score", "mean"))
    split = env["ENV"].str.split("|", expand=True)
    env["trading_platform"] = split[0]
    env["year"] = split[2].astype(int)
    groups, pairs = [], []
    for _, frame in env.groupby("trading_platform"):
        indices = frame.sort_values("year").index.to_numpy()
        groups.append(indices)
        pairs.extend(zip(indices[:-1], indices[1:]))
    if len(pairs) < 3:
        raise ValueError("纵向变化关联至少需要三条相邻年份变化")
    x = env["X"].to_numpy(dtype=float)
    y = env["score"].to_numpy(dtype=float)
    dx = np.array([x[j] - x[i] for i, j in pairs])
    dy = np.array([y[j] - y[i] for i, j in pairs])
    observed = _slope_per_point(dx, dy)
    rng, exceedances, completed = np.random.default_rng(seed), 0, 0
    for size in [10_000] * (draws // 10_000) + ([draws % 10_000] if draws % 10_000 else []):
        permuted = np.repeat(y[None, :], size, axis=0)
        for indices in groups:
            order = rng.random((size, len(indices))).argsort(axis=1)
            permuted[:, indices] = np.take_along_axis(permuted[:, indices], order, axis=1)
        permuted_dy = np.column_stack([permuted[:, j] - permuted[:, i] for i, j in pairs])
        slopes = np.array([_slope_per_point(dx, candidate) for candidate in permuted_dy])
        exceedances += int((np.abs(slopes) >= abs(observed) - 1e-15).sum())
        completed += size
    return {
        "target": target,
        "n_comment_rows": int(len(rows)),
        "n_ENV": int(len(env)),
        "n_adjacent_year_changes": int(len(pairs)),
        "slope_score_units_per_0_1_X": 0.1 * observed,
        "year_label_permutation_p": (exceedances + 1) / (completed + 1),
        "permutation_draws": completed,
        "seed": seed,
        "interpretation": "exploratory_platform_within_year_change_association_not_causal",
    }
