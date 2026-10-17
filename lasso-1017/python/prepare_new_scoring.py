"""把 lasso-1017 的 B站 2022 new 评论整理为可审计的 ENV、X、Y 评分表。"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

import analysis_helpers as h
import x_scoring


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "数据"
RAW = DATA / "01_原始数据"
OUT = DATA / "06_1017工作数据"
SOCIAL_PLATFORMS = ("B站",)
ALLOWED_ENVS = {"抖音|双11|2022", "拼多多|双11|2022"}
EXCLUDED_IDS = {
    "INCR20261014:N12B3_Q12:local018",
    "INCR20261014:N12B5_Q07:local006",
    "INCR20261014:NEUTRAL12_Q09:local004",
}
MERCHANT_RE = re.compile(r"京东|淘宝|天猫|拼多多|抖音")
EVENT_RE = re.compile(r"618|双\s*(?:11|十一)|11[.．]11", re.I)
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
CATEGORIES = tuple([f"C{i}" for i in range(1, 6)] + [f"R{i}" for i in range(1, 7)])

# 探索性补充：只计入目前已定位到的明确规则类别；不把 unknown 补为 yes，
# 不满足正式 X 的文献覆盖门槛，因此不会写入正式规则库或正式关联分析。
PARTIAL_RULE_EVIDENCE = {
    "抖音|双11|2021": {
        "C1": "https://www.sohu.com/a/492997242_284687",
        "C2": "https://www.sohu.com/a/492997242_284687",
        "R6": "https://www.sohu.com/a/492997242_284687",
    },
    "抖音|双11|2022": {
        "C1": "https://www.roundchic.com/h-nd-353.html",
        "C2": "https://www.roundchic.com/h-nd-353.html",
        "R1": "https://www.roundchic.com/h-nd-353.html",
        "R6": "https://www.roundchic.com/h-nd-353.html",
    },
    "抖音|双11|2023": {
        "C1": "https://www.cnecn.com.cn/hangye/lingshou/2023-09-19/13732.html",
        "C2": "https://www.cnecn.com.cn/hangye/lingshou/2023-09-19/13732.html",
        "C3": "https://www.cnecn.com.cn/hangye/lingshou/2023-09-19/13732.html",
        "R6": "https://www.cnecn.com.cn/hangye/lingshou/2023-09-19/13732.html",
    },
    "拼多多|双11|2021": {
        "C2": "https://m.maijiaw.com/article/499606",
        "C4": "https://m.maijiaw.com/article/499606",
        "C5": "https://m.maijiaw.com/article/499606",
        "R2": "https://m.maijiaw.com/article/499606",
    },
    "拼多多|双11|2022": {
        "C1": "https://www.itheat.com/view/34081.html",
        "C2": "https://www.itheat.com/view/34081.html",
        "C4": "https://www.itheat.com/view/34081.html",
        "R1": "https://www.itheat.com/view/34081.html",
    },
}


def first_match(pattern: re.Pattern[str], value: object) -> str | None:
    match = pattern.search(str(value or ""))
    return match.group() if match else None


def normalize_merchant(value: str) -> str:
    return "淘宝" if value == "天猫" else value


def normalize_event(value: str) -> str:
    return "618" if value == "618" else "双11"


def classify_env(row: pd.Series) -> pd.Series:
    """讨论文本/检索语的明确年份优先；没有时退回主题发表年份。"""
    discussion = " ".join(str(row.get(name) or "") for name in ("text", "topic_title", "topic_text", "query"))
    query_context = " ".join(str(row.get(name) or "") for name in ("query", "topic_title", "topic_text", "text"))
    merchant = first_match(MERCHANT_RE, query_context)
    event = first_match(EVENT_RE, query_context)
    explicit_year = first_match(YEAR_RE, discussion)
    if explicit_year:
        year, year_basis = int(explicit_year), "明确讨论年份"
    else:
        topic_time = str(row.get("topic_time_iso") or "")
        match = YEAR_RE.search(topic_time)
        if not match:
            return pd.Series({"ENV": None, "ENV_status": "unresolved_missing_topic_year", "year_basis": None})
        year, year_basis = int(match.group()), "主题发表年份"
    if not merchant or not event:
        return pd.Series({"ENV": None, "ENV_status": "unresolved_missing_platform_or_event", "year_basis": year_basis})
    merchant = normalize_merchant(merchant)
    event = normalize_event(event)
    return pd.Series({"ENV": f"{merchant}|{event}|{year}", "ENV_status": "resolved", "year_basis": year_basis})


def load_new_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for social_platform in SOCIAL_PLATFORMS:
        for path in sorted((RAW / social_platform / "new").glob("*.csv")):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame = frame.loc[frame["record_type"].eq("comment")].copy()
            frame["social_platform"] = social_platform
            frame["source_file"] = str(path.relative_to(ROOT))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("未找到 B站的 new CSV")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.loc[rows["text"].str.strip().ne("")].copy()
    native = rows["comment_id"].where(rows["comment_id"].str.strip().ne(""), rows["local_record_id"])
    rows["row_id"] = "NEW:" + rows["social_platform"] + ":" + rows["topic_id"] + ":" + native
    rows = rows.loc[~rows["row_id"].isin(EXCLUDED_IDS)].copy()
    rows = rows.drop_duplicates(subset=["social_platform", "topic_id", "comment_id", "text"], keep="first")
    return rows.reset_index(drop=True)


def score_y(rows: pd.DataFrame, target: str) -> np.ndarray:
    vectorizer, coef, intercept = h.load_scoring_model(target)
    with threadpool_limits(limits=1):
        matrix = vectorizer.transform(rows["text"].tolist())
        return expit(matrix @ coef + intercept)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = load_new_rows()
    rows[["ENV", "ENV_status", "year_basis"]] = rows.apply(classify_env, axis=1)
    rows = rows.loc[rows["ENV"].isin(ALLOWED_ENVS)].copy()
    _, env_scores = x_scoring.compute_x()
    x_lookup = env_scores.set_index("ENV")["X_ENV"]
    rows["X"] = rows["ENV"].map(x_lookup)
    partial_x = {env: len(categories) / len(CATEGORIES) for env, categories in PARTIAL_RULE_EVIDENCE.items()}
    needs_partial = rows["X"].isna() & rows["ENV"].isin(partial_x)
    rows.loc[needs_partial, "X"] = rows.loc[needs_partial, "ENV"].map(partial_x)
    rows["X_status"] = np.where(
        rows["ENV"].isna(),
        "missing_unresolved_ENV",
        np.where(
            needs_partial,
            "exploratory_partial_rule_count",
            np.where(rows["X"].notna(), "available_reviewed_rule_facts", "missing_no_reviewed_rule_facts"),
        ),
    )
    rows["Y1"] = score_y(rows, "Y1")
    rows["Y2"] = score_y(rows, "Y2")
    columns = [
        "row_id", "social_platform", "query", "topic_id", "topic_url", "topic_title",
        "topic_time_iso", "comment_time_iso", "text", "ENV", "ENV_status", "year_basis",
        "X", "X_status", "Y1", "Y2", "source_file",
    ]
    scored = rows[columns].sort_values(["social_platform", "topic_id", "row_id"]).reset_index(drop=True)
    output = OUT / "new_B站_2022_ENV_X_Y评分.csv"
    scored.to_csv(output, index=False, encoding="utf-8-sig")
    evidence_rows = [
        {"ENV": env, "category": category, "source_url": url,
         "method": "探索性：已定位规则类别数/11；未知类别不补计"}
        for env, categories in PARTIAL_RULE_EVIDENCE.items()
        for category, url in categories.items()
    ]
    pd.DataFrame(evidence_rows).to_csv(OUT / "探索性X_来源与类别.csv", index=False, encoding="utf-8-sig")
    print(f"写出：{output.relative_to(ROOT)}")
    print(f"new 评论：{len(scored)}")
    print("ENV 状态：")
    print(scored["ENV_status"].value_counts(dropna=False).to_string())
    print("X 状态：")
    print(scored["X_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
