"""旧版OLS + ENV聚类WCR11及CR2数学核心，来自冻结实现。"""
from __future__ import annotations
import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t
def _design(x):
    """Q has orthonormal intercept and centered-X columns, no normal-equation inverse."""
    u = np.asarray(x, dtype=float) / 0.1
    centered = u - u.mean()
    norm = float(np.linalg.norm(centered))
    q = np.column_stack((np.full(len(u), 1.0 / np.sqrt(len(u))), centered / norm))
    return q, norm, float(u.mean())

def _base_result(n: int, cfg: dict) -> dict:
    return {"status": "unavailable", "reasons": [], "warnings": [],
            "discovery_eligible": False, "quality_gate_failures": [],
            "n_input": n, "n_complete": 0, "n_missing": 0, "n_used": 0,
            "n_clusters": 0, "x_min": None, "x_max": None,
            "x_range": None, "x_distinct": 0, "score_range": None,
            "slope_score_units_per_0_1_x": None,
            "effect_score_percentage_points_per_0_1_x": None,
            "p_raw": None, "cr2_ci95_score_percentage_points": None,
            "cr2_se_score_units": None, "cr2_satterthwaite_df": None,
            "ci_test_disagree_at_0_05": None,
            "effect_description": cfg["estimand"]["effect_label"],
            "cluster_diagnostics": [], "leave_one_env": [],
            "bootstrap": {"status": "not_run"},
            "interpretation": "exploratory_model_score_association_not_causal_or_label_prevalence"}

def _analyze(score, x, env, cfg):
    y = np.asarray(score, dtype=float)
    x = np.asarray(x, dtype=float)
    env = list(env)
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x) or len(y) != len(env):
        raise ValueError("score, X, ENV must be equal-length row vectors")
    n = len(y)
    result = _base_result(n, cfg)
    valid_env = np.array([isinstance(v, str) and bool(v.strip()) for v in env])
    complete = np.isfinite(y) & np.isfinite(x) & valid_env
    result.update(n_complete=int(complete.sum()), n_missing=int((~complete).sum()))
    finite_x = x[np.isfinite(x)]
    if len(finite_x):
        result.update(x_min=float(finite_x.min()), x_max=float(finite_x.max()),
                      x_range=float(np.ptp(finite_x)), x_distinct=int(len(np.unique(finite_x))))
    if len(y[np.isfinite(y)]):
        result["score_range"] = float(np.ptp(y[np.isfinite(y)]))
    result["n_clusters"] = len(set(e for e, valid in zip(env, valid_env) if valid))
    if not complete.all():
        result["reasons"].append("incomplete_rows_no_silent_deletion")
        return result
    if n <= 2:
        result["reasons"].append("insufficient_rows")
        return result
    result["n_used"] = n
    policy = cfg["availability"]
    quality = cfg["quality_gates"]
    tol = policy["constant_relative_tolerance"]
    if np.ptp(y) <= tol * max(1.0, float(np.max(np.abs(y)))):
        result["reasons"].append("constant_or_numerically_constant_score")
    if np.ptp(x) <= tol * max(1.0, float(np.max(np.abs(x)))):
        result["reasons"].append("rank_deficient_or_numerically_constant_X")
    if result["reasons"]:
        return result
    groups = sorted(set(env))
    indices = [np.flatnonzero(np.asarray(env) == group) for group in groups]
    g = len(groups)
    q, norm, mean_u = _design(x)
    a = q.T @ y
    residual = y - q @ a
    slope = float(a[1] / norm)
    result.update(slope_score_units_per_0_1_x=slope,
                  effect_score_percentage_points_per_0_1_x=100.0 * slope,
                  intercept_score_units=float(a[0] / np.sqrt(n) - slope * mean_u))
    if g < policy["minimum_clusters_algebraic"]:
        result["reasons"].append("fewer_than_two_ENV_clusters")
    if g < quality["minimum_clusters"]:
        result["quality_gate_failures"].append("ENV_clusters_below_6")
    information = np.array([np.sum(q[ix, 1] ** 2) for ix in indices])
    effective_information = float(1.0 / np.sum(information ** 2))
    result["effective_information_clusters"] = effective_information
    if effective_information + 1e-12 < quality["minimum_effective_information_clusters"]:
        result["quality_gate_failures"].append("effective_X_information_clusters_below_4")
    if g < policy["diagnostic_warning_cluster_count_below"]:
        result["warnings"].append("few_ENV_clusters_bootstrap_not_guaranteed_reliable")
    if max(information) > policy["diagnostic_warning_max_information_share_above"]:
        result["warnings"].append("X_information_concentrated_in_ENV")
    adj_weights, norms, c_columns, cr0_scores, cr2_scores = [], [], [], [], []
    singular = False
    for group, ix, info in zip(groups, indices, information):
        qg = q[ix]
        u, s, _ = np.linalg.svd(qg, full_matrices=False)
        minimum_eigenvalue = float(1 - np.max(s) ** 2)
        cluster = {"ENV": group, "n": int(len(ix)), "row_share": float(len(ix)/n),
                   "x_min": float(x[ix].min()), "x_max": float(x[ix].max()),
                   "x_distinct": int(len(np.unique(x[ix]))),
                   "leverage_trace": float(np.sum(s ** 2)),
                   "leverage_max_eigenvalue": float(np.max(s) ** 2),
                   "residual_block_min_eigenvalue": minimum_eigenvalue,
                   "X_information_share": float(info), "CR2_variance_share": None}
        result["cluster_diagnostics"].append(cluster)
        cr0_scores.append(float(qg[:, 1] @ residual[ix] / norm))
        if minimum_eigenvalue <= policy["minimum_residual_block_eigenvalue"]:
            singular = True
            continue
        # A_g=(I-Q_g Q_g')^(-1/2), applied in its rank-at-most-two nontrivial subspace.
        l_g = qg[:, 1] / norm
        v_g = l_g + u @ ((1 / np.sqrt(1-s*s) - 1) * (u.T @ l_g))
        adj_weights.append(v_g)
        norms.append(float(v_g @ v_g))
        c_columns.append(qg.T @ v_g)
        cr2_scores.append(float(v_g @ residual[ix]))
    # Always report descriptive deletion estimates; never recompute p-values on deletions.
    for group, ix in zip(groups, indices):
        keep = np.ones(n, dtype=bool); keep[ix] = False
        omitted = {"omitted_ENV": group, "n_remaining": int(keep.sum()),
                   "effect_score_percentage_points_per_0_1_x": None,
                   "effect_change_percentage_points": None, "sign_flip": None,
                   "status": "unavailable"}
        if keep.sum() > 2 and np.ptp(x[keep]) > tol * max(1.0, float(np.max(np.abs(x[keep])))):
            u = x[keep]/0.1
            uc = u-u.mean()
            deleted_slope = float(uc @ (y[keep]-y[keep].mean()) / (uc @ uc))
            omitted.update(status="descriptive_only", effect_score_percentage_points_per_0_1_x=100*deleted_slope,
                           effect_change_percentage_points=100*(deleted_slope-slope),
                           sign_flip=bool(deleted_slope * slope < 0))
        result["leave_one_env"].append(omitted)
    if singular:
        result["reasons"].append("singular_or_near_singular_cluster_residual_block")
        return result
    # p_g = (I-QQ') E_g A_g l_g. Disjoint E_g gives this G x G Gram matrix.
    c = np.column_stack(c_columns)
    gram = np.diag(norms) - c.T @ c
    trace = float(np.trace(gram))
    frobenius2 = float(np.sum(gram ** 2))
    df = trace ** 2 / frobenius2 if frobenius2 > 0 else float("nan")
    result["cr2_satterthwaite_df"] = float(df) if np.isfinite(df) else None
    v_cr2 = float(np.sum(np.square(cr2_scores)))
    scale2 = float(np.var(y, ddof=0) / (norm * norm))
    minimum_variance = policy["variance_relative_tolerance"] * scale2
    if not np.isfinite(df) or df <= 0:
        result["reasons"].append("undefined_CR2_Satterthwaite_df")
    elif df + 1e-12 < quality["minimum_cr2_satterthwaite_df"]:
        result["quality_gate_failures"].append("CR2_Satterthwaite_df_below_2")
    if np.isfinite(df) and df < policy["diagnostic_warning_cr2_df_below"]:
        result["warnings"].append("very_low_CR2_Satterthwaite_df")
    if not np.isfinite(v_cr2) or v_cr2 <= minimum_variance:
        result["reasons"].append("degenerate_CR2_slope_variance")
    else:
        shares = np.square(cr2_scores)/v_cr2
        for cluster, share in zip(result["cluster_diagnostics"], shares):
            cluster["CR2_variance_share"] = float(share)
        result["effective_CR2_variance_clusters"] = float(1/np.sum(shares**2))
        result["cr2_se_score_units"] = float(np.sqrt(v_cr2))
    if g > 1:
        cr1_factor = g/(g-1) * (n-1)/(n-2)
        v_cr1 = cr1_factor * float(np.sum(np.square(cr0_scores)))
    else:
        cr1_factor, v_cr1 = float("nan"), float("nan")
    if not np.isfinite(v_cr1) or v_cr1 <= minimum_variance:
        result["reasons"].append("degenerate_CR1S_slope_variance")
    result["cr1s_se_score_units"] = float(np.sqrt(v_cr1)) if v_cr1 > 0 else None
    if result["reasons"]:
        return result
    ci = 100 * (slope + np.array([-1.,1.]) * student_t.ppf(0.975, df) * np.sqrt(v_cr2))
    result["cr2_ci95_score_percentage_points"] = [float(v) for v in ci]
    result["cr2_t_statistic"] = float(slope / np.sqrt(v_cr2))
    result["cr2_p_descriptive_separate_from_primary"] = float(2*student_t.sf(abs(slope)/np.sqrt(v_cr2), df))
    bootstrap = _wild_bootstrap(y, q, norm, indices, slope/np.sqrt(v_cr1),
                                cr1_factor, minimum_variance, cfg)
    result["bootstrap"] = bootstrap
    if bootstrap["status"] != "available":
        result["reasons"].append("degenerate_bootstrap_studentization")
        result["status"] = "primary_p_unavailable_CR2_interval_available"
        return result
    result["p_raw"] = bootstrap["p_value"]
    result["status"] = "available" if not result["quality_gate_failures"] else "diagnostic_only_below_quality_gates"
    result["discovery_eligible"] = not result["quality_gate_failures"]
    result["ci_test_disagree_at_0_05"] = bool((bootstrap["p_value"] <= 0.05) != (ci[0] > 0 or ci[1] < 0))
    if result["ci_test_disagree_at_0_05"]:
        result["warnings"].append("primary_WCR_test_and_separate_CR2_interval_disagree_at_0_05")
    return result

def _sign_batches(g, cfg):
    boot = cfg["wild_cluster_bootstrap"]
    exact = g <= boot["exact_enumeration_max_clusters"]
    total = 2**g if exact else boot["monte_carlo_draws"]
    rng = np.random.Generator(np.random.PCG64(boot["seed"]))
    for start in range(0, total, boot["batch_size"]):
        size = min(boot["batch_size"], total-start)
        if exact:
            bits = ((np.arange(start,start+size,dtype=np.uint64)[:,None] >>
                     np.arange(g,dtype=np.uint64)[None,:]) & 1)
            yield 2*bits.astype(float)-1
        else:
            yield 2*rng.integers(0,2,size=(size,g),dtype=np.int64).astype(float)-1

def _wild_bootstrap(y, q, norm, indices, observed_t, cr1_factor, min_variance, cfg):
    g = len(indices)
    restricted_residual = y-y.mean()
    cluster_qtu = np.array([q[ix].T @ restricted_residual[ix] for ix in indices])
    cluster_slope_score = cluster_qtu[:,1]/norm
    projection_rows = np.array([(q[ix].T @ q[ix])[1,:]/norm for ix in indices])
    exceedances, bad, total = 0, 0, 0
    setting = cfg["wild_cluster_bootstrap"]
    threshold = abs(observed_t) - setting["tie_atol"] - setting["tie_rtol"]*abs(observed_t)
    for w in _sign_batches(g, cfg):
        bootstrap_coef_q = w @ cluster_qtu
        bootstrap_slope = bootstrap_coef_q[:,1]/norm
        cluster_residual_scores = w*cluster_slope_score - bootstrap_coef_q @ projection_rows.T
        variances = cr1_factor*np.sum(cluster_residual_scores**2,axis=1)
        valid = np.isfinite(variances) & (variances > min_variance)
        bad += int((~valid).sum())
        t = np.abs(bootstrap_slope[valid])/np.sqrt(variances[valid])
        exceedances += int(np.sum(t >= threshold))
        total += len(w)
    exact = g <= setting["exact_enumeration_max_clusters"]
    answer = {"status": "available" if bad == 0 else "unavailable",
              "variant": "WCR11", "observed_t_CR1S": float(observed_t),
              "enumeration": exact, "draws": total, "exceedances": exceedances,
              "degenerate_draws": bad, "p_value": None,
              "seed": None if exact else setting["seed"],
              "minimum_nonzero_p": 2/(2**g) if exact else 1/(total+1),
              "ties": "inclusive_with_frozen_absolute_and_relative_tolerance",
              "monte_carlo_tail_probability_CI95": None,
              "monte_carlo_standard_error": None}
    if not bad:
        p = exceedances/total if exact else (exceedances+1)/(total+1)
        answer["p_value"] = float(p)
        if not exact:
            lower = 0.0 if exceedances == 0 else beta_distribution.ppf(0.025,exceedances,total-exceedances+1)
            upper = 1.0 if exceedances == total else beta_distribution.ppf(0.975,exceedances+1,total-exceedances)
            answer["monte_carlo_tail_probability_CI95"] = [float(lower),float(upper)]
            answer["monte_carlo_standard_error"] = float(np.sqrt(p*(1-p)/(total+1)))
    return answer
