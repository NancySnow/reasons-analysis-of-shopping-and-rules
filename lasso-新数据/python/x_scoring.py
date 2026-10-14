"""从已核的规则事实及冻结元数据重新计算冻结新版 X。"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import pandas as pd

CATEGORIES = tuple([f'C{i}' for i in range(1, 6)] + [f'R{i}' for i in range(1, 7)])
MODULES = ('D1', 'D2', 'D3', 'D4')
DATA_DIR = Path(__file__).resolve().parent.parent / '数据' / '02_X匹配数据'


def _read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _json(value, default):
    return json.loads(value) if value else default


def _yes(value):
    if value not in ('True', 'False'):
        raise ValueError(f'应为 True/False，实际为 {value!r}')
    return value == 'True'


def document_gate(coverage):
    values = [coverage[k] for k in MODULES]
    if any(v not in (0, 1, 2) for v in values):
        raise ValueError('D1–D4 覆盖必须是已核的 0/1/2')
    return values[0] == 2 and sum(v >= 1 for v in values) >= 3 and sum(v == 2 for v in values) >= 2


def score_recorded(facts, *, coverage, explicit_absence=(), category_conflicts=()):
    # 与旧版 v5 score_recorded 相同：按语义事实去重，11 类等权。
    facts = {f['semantic_fact_id']: f for f in facts}.values()
    positive = {f['category'] for f in facts}
    absent = set(explicit_absence)
    conflicts = set(category_conflicts) | (positive & absent)
    if (positive | absent | conflicts) - set(CATEGORIES):
        raise ValueError('出现 11 类之外的类别')
    states = {c: 'conflict' if c in conflicts else 'observed_yes' if c in positive else 'explicit_no' if c in absent else 'unknown' for c in CATEGORIES}
    yes = sum(v == 'observed_yes' for v in states.values())
    uncertain = sum(v in ('unknown', 'conflict') for v in states.values())
    gate = document_gate(coverage)
    return dict(states=states, X=yes / 11 if gate and not conflicts else None,
                observed_category_count=yes, unknown_or_conflict_count=uncertain,
                document_gate_pass=gate, coverage_rate=sum(coverage.values()) / 8,
                missing_evidence_lower=yes / 11,
                missing_evidence_upper=(yes + uncertain) / 11,
                condition_subscore=sum(states[c] == 'observed_yes' for c in CATEGORIES[:5]) / 5,
                relation_subscore=sum(states[c] == 'observed_yes' for c in CATEGORIES[5:]) / 6)


def compose(env_facts, topic_facts=(), *, explicit_exclude_ids=(), conflict_ids=()):
    known = {f['semantic_fact_id']: f for f in env_facts}
    extra = {f['semantic_fact_id']: f for f in topic_facts}
    excluded = set(explicit_exclude_ids)
    if excluded - set(known):
        raise ValueError('主题排除指向未知 ENV 语义事实')
    conflicts = set(conflict_ids)
    for key, fact in extra.items():
        if key in known and fact['category'] != known[key]['category']:
            conflicts.add(key)
    if conflicts:
        raise ValueError(f'主题事实存在尚未处置的冲突：{sorted(conflicts)}')
    merged = {key: value for key, value in known.items() if key not in excluded}
    merged.update(extra)
    return list(merged.values())


def compute_x(data_dir=None, check_reference=True):
    """返回 (全评论 X 数据框, 28 个 ENV 数据框)。冻结 X 只在计算后用于验收。

    平台/活动/年份与来源资格包含已固定的人工核对，不能仅由评论文本自动推定。
    未获得冻结新版 X 联接资格的记录保持缺失；不因新规则表覆盖它而补填。
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    environment_rows = _read(data_dir / 'environment_metadata.csv')
    environment = {r['ENV']: r for r in environment_rows}
    if len(environment) != len(environment_rows):
        raise ValueError('重复 ENV')
    facts = defaultdict(list)
    for row in _read(data_dir / 'rule_facts.csv'):
        if row['ENV'] not in environment or row['evidence_status'] != 'observed_yes':
            raise ValueError('规则事实的 ENV 或状态未获准')
        facts[row['ENV']].append(row)
    if len({f['semantic_fact_id'] for rows in facts.values() for f in rows}) != sum(map(len, facts.values())):
        raise ValueError('规则表 semantic_fact_id 重复')
    topic_reviews = {(r['topic_native_id'], r['ENV']): r for r in _read(data_dir / 'topic_rule_reviews.csv')}
    topic_facts = defaultdict(list)
    for r in _read(data_dir / 'topic_rule_facts.csv'):
        topic_facts[(r['topic_native_id'], r['ENV'])].append(r)
    env_scored = {}; env_result = []
    for key, e in environment.items():
        coverage = {d: int(e[d]) for d in MODULES}
        absence = [c for c in CATEGORIES if e[c + '_state_frozen'] == 'explicit_no']
        conflicts = [c for c in CATEGORIES if e[c + '_state_frozen'] == 'conflict']
        s = score_recorded(facts[key], coverage=coverage, explicit_absence=absence, category_conflicts=conflicts)
        if not _yes(e['source_admission_pass']):
            s['X'] = None
        if check_reference and any(s['states'][c] != e[c + '_state_frozen'] for c in CATEGORIES):
            raise ValueError(f'{key} 类别状态与冻结记录不同')
        env_scored[key] = s
        env_result.append({'ENV': key, 'trading_platform': e['trading_platform'], 'event': e['event'],
                           'event_year': int(e['event_year']), 'X_ENV': s['X'],
                           'observed_category_count': s['observed_category_count'],
                           'unknown_category_count': s['unknown_or_conflict_count'],
                           'document_gate_pass': s['document_gate_pass'],
                           'source_admission_pass': _yes(e['source_admission_pass']),
                           'coverage_rate': s['coverage_rate'],
                           'condition_subscore': s['condition_subscore'],
                           'relation_subscore': s['relation_subscore'],
                           **coverage, **{c: s['states'][c] for c in CATEGORIES}})
    metadata = _read(data_dir / 'comment_X_metadata.csv'); rows = []
    if len({r['row_id'] for r in metadata}) != len(metadata):
        raise ValueError('重复评论 row_id')
    for row in metadata:
        key = row['ENV'] or None; value = None; mode = 'not_approved_missing'
        if _yes(row['approved_X_row']):
            if key not in environment or env_scored[key]['X'] is None:
                raise ValueError(f"已批准评论没有可用 ENV：{row['row_id']}")
            selected = facts[key]
            if row['basis'] == 'evidence_based_union_recode':
                topic_key = (row['topic_id'].split('|')[-1], key)
                if topic_key not in topic_reviews:
                    raise ValueError(f'缺少旧主题修正：{topic_key}')
                t = topic_reviews[topic_key]
                selected = compose(selected, topic_facts[topic_key],
                                   explicit_exclude_ids=_json(t['excluded_semantic_fact_ids_json'], []),
                                   conflict_ids=_json(t['conflict_ids_json'], []))
                mode = 'ENV_plus_reviewed_topic_facts'
            elif row['basis'] == 'ENV_default_inheritance_assumption':
                mode = 'ENV_default_inheritance_assumption'
            elif row['basis'] == 'ENV_menu_metadata_proxy':
                if row.get('data_origin') != 'neutral_increment_frozen':
                    raise ValueError('新增菜单代理必须属于冻结新增来源')
                mode = 'reviewed_increment_campaign_ENV_menu_proxy'
            elif row['stream'] == 'A' and not row['basis']:
                mode = 'approved_ENV_join_original_A_basis_not_exported'
            else:
                raise ValueError(f"未知冻结 X 计算依据：{row['row_id']}")
            e = environment[key]
            score = score_recorded(selected, coverage={d: int(e[d]) for d in MODULES},
                                   explicit_absence=[c for c in CATEGORIES if e[c + '_state_frozen'] == 'explicit_no'],
                                   category_conflicts=[c for c in CATEGORIES if e[c + '_state_frozen'] == 'conflict'])
            value = score['X']
            saved_states = _json(row['category_states_frozen_json'], None)
            if check_reference and saved_states is not None and saved_states != score['states']:
                raise ValueError(f"冻结评论类别状态不同：{row['row_id']}")
        rows.append({'row_id': row['row_id'], 'stream': row['stream'], 'topic_id': row['topic_id'],
                     'family_id': row['family_id'], 'X': value, 'ENV': key,
                     'X_missing_reason': row['X_missing_reason'] or None,
                     'basis_original': row['basis'] or None, 'recompute_mode': mode,
                     'data_origin': row.get('data_origin'), 'original_family_id': row.get('original_family_id') or None})
    if check_reference:
        # 此文件只在所有计算完成之后读取，不为规则计数、联接或资格提供数值。
        reference = _read(data_dir / 'frozen_X_reference.csv')
        if [r['row_id'] for r in rows] != [r['row_id'] for r in reference]:
            raise ValueError('验收记录集合或顺序变化')
        for actual, saved in zip(rows, reference):
            expected = float(saved['X_saved']) if saved['X_saved'] else None
            if actual['X'] != expected or actual['ENV'] != (saved['ENV_saved'] or None):
                raise ValueError(f"X 或 ENV 与冻结版本不一致：{actual['row_id']}")
    return pd.DataFrame(rows), pd.DataFrame(env_result)
