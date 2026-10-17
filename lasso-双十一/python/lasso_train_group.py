"""旧route3数学流程的可移植封装；实际拟合与历史冻结结果读取分开。"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"): os.environ[_key]="1"
import csv,hashlib,json,re,sys,time,unicodedata
from pathlib import Path
import numpy as np
import scipy,sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from threadpoolctl import threadpool_info,threadpool_limits
ROUTES=("direct_context","lasso_surface","lasso_context")
LASSO_ROUTES=("lasso_surface","lasso_context")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    """Atomic metadata replacement; large arrays are written separately."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def csvout(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        if not rows:
            raise ValueError('An empty CSV needs explicit fields')
        fields = list(rows[0])
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)

def norm(text):
    # Exact predecessor preflight.norm behavior.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).lower()).strip()

def lambda_selection(grid, losses, ns):
    # Exact predecessor pipeline_guards.lambda_selection arithmetic and ties.
    ls = np.asarray(grid, float)
    loss = np.asarray(losses, float)
    n = np.asarray(ns, float)
    if (len(set(ls)) != len(ls) or (ls <= 0).any()
            or loss.shape != (len(ls), len(n)) or (n <= 0).any()
            or len(n) < 2 or not np.isfinite(loss).all()):
        raise ValueError('Invalid CV table')
    weights = n / n.sum()
    means = loss @ weights
    se = np.sqrt(((loss - means[:, None]) ** 2 @ weights) / (len(n) - 1))
    best = np.flatnonzero(means == means.min())
    j = int(best[np.argmax(ls[best])])
    eligible = means <= means[j] + se[j]
    return {
        'lambda_min': float(ls[j]), 'lambda_1se': float(ls[eligible].max()),
        'mean_logloss': means.tolist(), 'SE_logloss': se.tolist(),
        'mean_binomial_deviance': (2 * means).tolist(), 'SE_binomial_deviance': (2 * se).tolist(),
        'threshold_logloss': float(means[j] + se[j]),
        'SE_is_CV_selection_heuristic_not_CI': True,
    }

def _ids(plan, key):
    values = plan.get(key)
    if not isinstance(values, list) or any(not isinstance(i, str) for i in values) or len(values) != len(set(values)):
        raise ValueError('Invalid or repeated IDs: ' + key)
    return set(values)

def validate_plan(plan, families, cohort_ids, route, target, *, synthetic=False):
    if plan.get('route') != route or plan.get('target') != target:
        raise ValueError('Plan route/target mismatch')
    if type(plan.get('outer_seed')) is not int or type(plan.get('inner_seed')) is not int:
        raise ValueError('Explicit predeclared integer seeds required')
    if not synthetic:
        expected_seed = 20261013 + 1000 * (ROUTES.index(route) + 1) + int(target[1:])
        if plan['outer_seed'] != expected_seed or plan['inner_seed'] != expected_seed + 100000:
            raise ValueError('Predeclared one-shot seed formula mismatch')
        if plan.get('allocation_attempts') != 1 or plan.get('split_search_performed') is not False:
            raise ValueError('Exactly one allocation, without split search, is required')
    train, test = _ids(plan, 'train_ids'), _ids(plan, 'test_ids')
    if plan.get('preparation_support_status') == 'UNSUPPORTED_OUTER_SPLIT':
        if train or test or plan.get('inner_folds') != [] or not plan.get('unsupported_reasons'):
            raise ValueError('Unsupported outer split must contain no partial partition and a reason')
        return
    if not train or not test or train & test or train | test != set(cohort_ids):
        raise ValueError('Outer train/test must partition the entire H+A cohort')
    if {families[i] for i in train} & {families[i] for i in test}:
        raise ValueError('Outer family leakage')
    folds = plan.get('inner_folds')
    if route == 'direct_context':
        if folds != []:
            raise ValueError('Direct context must have no tuning folds')
        return
    if folds == [] and plan.get('preparation_support_status') == 'UNSUPPORTED_TRAINING' and plan.get('unsupported_reasons'):
        return
    if not isinstance(folds, list) or len(folds) != 5 or {f.get('fold') for f in folds} != set(range(5)):
        raise ValueError('Exactly five unique inner folds required')
    validation = []
    for fold in folds:
        fit, valid = _ids(fold, 'fit_ids'), _ids(fold, 'validation_ids')
        if not fit or not valid or fit & valid or fit | valid != train:
            raise ValueError('Each fold must partition only its own outer training rows')
        if {families[i] for i in fit} & {families[i] for i in valid}:
            raise ValueError('Inner family leakage')
        validation.extend(valid)
    if len(validation) != len(train) or set(validation) != train:
        raise ValueError('Each outer-training row must validate exactly once')

def support_report(lookup, plan, families, label_key, target):
    def counts(ids):
        positive = [i for i in ids if lookup[i][label_key][target] == 1]
        negative = [i for i in ids if lookup[i][label_key][target] == 0]
        return {'N': len(ids), 'positive': len(positive), 'negative': len(negative),
                'positive_families': len({families[i] for i in positive}),
                'negative_families': len({families[i] for i in negative})}
    train = counts(plan['train_ids'])
    folds = [{'fold': f['fold'], 'fit': counts(f['fit_ids']),
              'validation': counts(f['validation_ids'])} for f in plan['inner_folds']]
    supported = (train['positive_families'] >= 5 and train['negative_families'] >= 5
                 and len(folds) == 5 and all(min(f[k]['positive'], f[k]['negative']) >= 1
                 for f in folds for k in ('fit', 'validation')))
    return {'train': train, 'folds': folds,
            'minimum_CV_support_met': bool(supported),
            'gate_uses_training_only': True,
            'low_positive_support': train['positive'] < 25 or any(f['validation']['positive'] < 5 for f in folds)}

def vectorize(lookup, ids, folder):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), min_df=2,
                                max_df=1.0, norm='l2', preprocessor=norm,
                                lowercase=False, sublinear_tf=False)
    matrix = vectorizer.fit_transform([lookup[i]['text'] for i in ids])
    dump(folder / 'vocabulary_idf.json', {
        'fit_ids': ids, 'vocabulary': {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        'idf': vectorizer.idf_.tolist(), 'feature_N': matrix.shape[1],
        'vectorizer_parameters': {'analyzer': 'char', 'ngram_range': [2, 3], 'min_df': 2,
                                  'norm': 'l2', 'lowercase': False, 'sublinear_tf': False},
    })
    return vectorizer, matrix

def metrics(y, probability):
    y, p = np.asarray(y, int), np.asarray(probability, float)
    n, positives = len(y), int(y.sum())
    if not n:
        return {'N': 0, 'status': 'UNAVAILABLE_EMPTY_SUBSET', 'constant_score': None}
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability')
    ll, brier = float(log_loss(y, p, labels=[0, 1])), float(brier_score_loss(y, p))
    return {'N': n, 'positive_N': positives, 'log_loss': ll,
            'mean_binomial_deviance': 2 * ll, 'Brier': brier, 'RMSE': float(np.sqrt(brier)),
            'AP': float(average_precision_score(y, p)) if positives else None,
            'ROC_AUC': float(roc_auc_score(y, p)) if 0 < positives < n else None,
            'constant_score': bool(np.ptp(p) < 1e-12), 'score_min': float(p.min()),
            'score_max': float(p.max()),
            'scores_are_weak_label_predictions_not_behavior_probabilities': True}

def terminal(folder, status, **details):
    result = {'status': status, 'terminal': True, **details}
    result['artifact_sha256'] = {
        str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name not in {'COMPLETE.json', 'UNSUPPORTED.json', 'FAILED.json', 'CONSTANT_TRAIN_LABEL.json'}
    }
    marker = 'COMPLETE.json' if status == 'COMPLETE' else status + '.json'
    dump(folder / marker, result)
    return result

def fit_route(lookup, plan, families, route, target, out, solver, *,
              synthetic=False, synthetic_grid=None, refresh=lambda: None, guard=None):
    if route not in LASSO_ROUTES:
        raise ValueError('Only LASSO routes can call fit_route')
    if synthetic_grid is not None and not synthetic:
        raise ValueError('Real fits cannot override the lambda grid')
    binding = {"execution_mode": "portable_real_retraining"}
    if synthetic or guard is not None:
        raise ValueError("This portable entry point accepts real frozen data only")
    validate_plan(plan, families, lookup, route, target, synthetic=synthetic)
    folder = out / route / target
    # 长训练若被终端窗口打断，可从同一份、未改动的训练输入续跑。
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / 'split_id_order.json', plan)
    if plan.get('preparation_support_status') == 'UNSUPPORTED_OUTER_SPLIT':
        return terminal(folder, 'UNSUPPORTED', route=route, target=target, synthetic_only=synthetic, **binding,
                        fits=0, reason=plan['unsupported_reasons'], predictions_available=False)
    label_key = 'labels_surface' if route == 'lasso_surface' else 'labels_context'
    support = support_report(lookup, plan, families, label_key, target)
    dump(folder / 'support.json', support)
    common = {'route': route, 'target': target, 'synthetic_only': synthetic, **binding,
              'outer_train_N': len(plan['train_ids']), 'outer_test_N': len(plan['test_ids'])}
    if not min(support['train']['positive'], support['train']['negative']):
        return terminal(folder, 'CONSTANT_TRAIN_LABEL', **common, fits=0,
                        reason='Outer training labels have one class; no pseudo-fit or split retry',
                        predictions_available=False)
    if not support['minimum_CV_support_met'] or plan.get('preparation_support_status') == 'UNSUPPORTED_TRAINING':
        return terminal(folder, 'UNSUPPORTED', **common, fits=0,
                        reason='Predeclared training/fold class and family gate failed; no split retry',
                        preparation_reasons=plan.get('unsupported_reasons', []),
                        predictions_available=False)
    logs, cv_rows = [], []
    try:
        refresh()
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(p['num_threads'] != 1 for p in pools):
                raise RuntimeError('BLAS/OpenMP single-thread constraint not effective')
            dump(folder / 'runtime.json', {'python': sys.version, 'numpy': np.__version__,
                'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'threadpools': pools})

            def fit(matrix, ids, lam, destination, stage):
                if destination.exists():
                    saved = np.load(destination, allow_pickle=False)
                    weight = np.asarray(saved['coef'][0], float)
                    intercept = float(saved['intercept'][0])
                    diagnostic = solver.kkt(matrix, np.asarray([lookup[i][label_key][target] for i in ids]),
                                            weight, intercept, lam)
                    diagnostic.update(solver='reused_interrupted_fit', iterations=0,
                                      converged=diagnostic['max_KKT'] <= 1e-7)
                    if not diagnostic['converged']:
                        raise RuntimeError('Interrupted fit failed KKT gate')
                    return solver.CheckedModel(weight, intercept, 0, diagnostic)
                started = time.monotonic()
                model = solver.fit_checked(matrix, np.asarray([lookup[i][label_key][target] for i in ids]),
                                           lam, tolerance=1e-7)
                if (not model.diagnostic.get('converged')
                        or not np.isfinite(model.diagnostic['max_KKT'])
                        or model.diagnostic['max_KKT'] > 1e-7):
                    raise RuntimeError('Solver KKT/convergence gate failed')
                np.savez_compressed(destination, coef=model.coef_, intercept=model.intercept_, classes=model.classes_)
                logs.append({'stage': stage, 'lambda': float(lam), 'fit_N': len(ids),
                             'feature_N': matrix.shape[1], 'nonzero': int(np.count_nonzero(model.coef_)),
                             'seconds': time.monotonic() - started, **model.diagnostic,
                             'artifact': str(destination.relative_to(folder)), 'sha256': sha(destination)})
                dump(folder / 'fit_log.json', logs)
                return model

            cache = []
            for fold in sorted(plan['inner_folds'], key=lambda f: f['fold']):
                fold_dir = folder / f"fold{fold['fold']}"
                fold_dir.mkdir(exist_ok=True)
                ids, valid = list(fold['fit_ids']), list(fold['validation_ids'])
                vectorizer, matrix = vectorize(lookup, ids, fold_dir)
                valid_matrix = vectorizer.transform([lookup[i]['text'] for i in valid])
                dump(fold_dir / 'id_order.json', {'fit_ids': ids, 'validation_ids': valid})
                cache.append((fold['fold'], fold_dir, ids, valid, matrix, valid_matrix))
            grid = list(10.0 ** np.arange(-6, -.999, .5)) if synthetic_grid is None else sorted(synthetic_grid)
            scores, nonzeros = {}, {}
            extended_low, extended_high = False, False
            while True:
                for lam in sorted(set(grid) - set(scores), reverse=True):
                    scores[lam], nonzeros[lam] = [], []
                    for k, fold_dir, ids, valid, matrix, valid_matrix in cache:
                        model = fit(matrix, ids, lam, fold_dir / f'lambda_{lam:.12g}.npz', f'CV{k}')
                        p = model.predict_proba(valid_matrix)[:, 1]
                        y = [lookup[i][label_key][target] for i in valid]
                        loss = float(log_loss(y, np.clip(p, 1e-5, 1-1e-5), labels=[0, 1]))
                        scores[lam].append(loss)
                        nonzeros[lam].append(int(np.count_nonzero(model.coef_)))
                        cv_rows.append({'route': route, 'target': target, 'fold': k, 'lambda': float(lam),
                                        'validation_N': len(valid), 'mean_logloss_clipped_1e5': loss,
                                        'unclipped_logloss': float(log_loss(y, p, labels=[0, 1])),
                                        'mean_binomial_deviance': 2 * loss})
                        csvout(folder / 'cv_fold_losses.csv', cv_rows)
                grid = sorted(scores)
                selection = lambda_selection(grid, [scores[lam] for lam in grid], [len(c[3]) for c in cache])
                new = []
                if not synthetic:
                    if selection['lambda_min'] == grid[0] and not extended_low:
                        new.extend([grid[0] / 10, grid[0] / np.sqrt(10)])
                        extended_low = True
                    if (selection['lambda_min'] == grid[-1]
                            or (selection['lambda_1se'] == grid[-1] and max(nonzeros[grid[-1]]) > 0)) and not extended_high:
                        new.extend([grid[-1] * np.sqrt(10), grid[-1] * 10])
                        extended_high = True
                if not new:
                    break
                grid = sorted(set(grid + new))
            selection.update(route=route, target=target, grid=grid, extended_lower=extended_low,
                             extended_upper=extended_high, primary='lambda_1se',
                             lambda_min_role='diagnostic_only', selection_uses_only_outer_training_folds=True)
            dump(folder / 'selection.json', selection)
            selection_hash = sha(folder / 'selection.json')
            refresh()
            full = folder / 'full_train_path'
            full.mkdir(exist_ok=True)
            train_ids = list(plan['train_ids'])
            vectorizer, matrix = vectorize(lookup, train_ids, full)
            weights, intercepts, path_rows, primary = [], [], [], None
            for lam in sorted(grid, reverse=True):
                model = fit(matrix, train_ids, lam, full / f'lambda_{lam:.12g}.npz', 'full_train_path')
                weights.append(model.coef_[0])
                intercepts.append(model.intercept_[0])
                path_rows.append({'lambda': float(lam), 'nonzero': int(np.count_nonzero(model.coef_))})
                if lam == selection['lambda_1se']:
                    primary = model
            if primary is None or sha(folder / 'selection.json') != selection_hash:
                raise RuntimeError('Selected lambda changed or selected model missing')
            np.savez_compressed(folder / 'coefficient_path.npz',
                                lambda_grid=np.asarray(sorted(grid, reverse=True)),
                                coefficients=np.asarray(weights), intercepts=np.asarray(intercepts))
            csvout(folder / 'path_summary.csv', path_rows)
            # First holdout transform occurs AFTER frozen selection and full training.
            test_ids = list(plan['test_ids'])
            test_matrix = vectorizer.transform([lookup[i]['text'] for i in test_ids])
            p = primary.predict_proba(test_matrix)[:, 1]
            y = np.asarray([lookup[i][label_key][target] for i in test_ids])
            dump(folder / 'test_metrics.json', metrics(y, p))
            predictions = [{
                'row_id': row_id, 'test_order': j, 'stream': lookup[row_id]['stream'],
                'family_id': families[row_id], 'route': route, 'target': target,
                'raw_label': int(y[j]), 'raw_surface_label': lookup[row_id]['labels_surface'][target],
                'raw_context_label': lookup[row_id]['labels_context'][target],
                'probability': float(p[j]), 'lambda_1se': selection['lambda_1se'],
                'prediction_role': 'entire_independent_pooled_40pct_holdout', 'synthetic_only': synthetic,
            } for j, row_id in enumerate(test_ids)]
            csvout(folder / 'test_predictions.csv', predictions)
            features = vectorizer.get_feature_names_out()
            csvout(folder / 'nonzero_coefficients.csv', [
                {'feature': str(features[j]), 'coefficient': float(primary.coef_[0, j])}
                for j in np.flatnonzero(primary.coef_[0])], ['feature', 'coefficient'])
            dump(folder / 'intercept.json', {'intercept': float(primary.intercept_[0]),
                                             'nonzero_N': int(np.count_nonzero(primary.coef_))})
            refresh()
            return terminal(folder, 'COMPLETE', **common, fits=len(logs),
                            selection_sha256=selection_hash,
                            total_fit_seconds=sum(log['seconds'] for log in logs),
                            prediction_sha256=sha(folder / 'test_predictions.csv'),
                            metrics_sha256=sha(folder / 'test_metrics.json'))
    except Exception as error:
        terminal(folder, 'FAILED', **common, fits=len(logs), error=repr(error), no_valid_completed_model=True)
        raise
