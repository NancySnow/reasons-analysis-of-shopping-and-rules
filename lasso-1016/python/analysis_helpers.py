"""冻结新版复现入口。所有输入来自本项目；new目录不会自动进入分析。"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json,csv,warnings,platform
from datetime import datetime
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
import lasso_train_group as training
import lasso_solver
import y_labeler
import cluster_inference_p

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'数据'
TARGETS={'Y1':('y06','Y1_多花','04_X与Y1关联分析数据'),'Y2':('y10','Y2_求助','05_X与Y2关联分析数据')}

def runtime_check():
    actual={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__}
    reference={'python':'3.12.14','numpy':'2.3.5','scipy':'1.17.0','scikit-learn':'1.8.0'}
    if actual!=reference:
        warnings.warn('当前版本与冻结执行环境不同。代码可运行性与数值复现应分别核对；优先使用Python 3.12及requirements.txt，评分允许浮点微差。')
    return actual

def target_folder(name):
    return DATA/'03_Y模型数据'/TARGETS[name][1]

def load_target(name):
    target=TARGETS[name][0]; folder=target_folder(name)
    df=pd.read_csv(folder/'建模全池.csv',keep_default_na=False,dtype={'row_id':str,'family_id':str,'text':str,'topic_id':str})
    plan=json.loads((folder/'冻结split.json').read_text(encoding='utf-8'))
    families=dict(zip(df.row_id,df.family_id))
    if df.row_id.duplicated().any():raise ValueError('row_id重复')
    training.validate_plan(plan,families,set(df.row_id),'lasso_context',target)
    tr=set(plan['train_ids']); va={i:f['fold']for f in plan['inner_folds']for i in f['validation_ids']}
    for r in df.itertuples(index=False):
        if r.split!=('train'if r.row_id in tr else'test'):raise ValueError('CSV和冻结split不一致')
        if str(r.cv_validation_fold)!=str(va.get(r.row_id,'')):raise ValueError('CV标记不一致')
    return df,plan

def check_weak_labels(name,df=None):
    """真正执行冻结规则，逐条核对弱标签；不重新定义构念。"""
    if df is None:df,_=load_target(name)
    target=TARGETS[name][0]
    labels=np.array([y_labeler.label_text(t)['outcome_labels'][target]for t in df.text],dtype=int)
    expected=df.weak_label.to_numpy(dtype=int)
    bad=np.flatnonzero(labels!=expected)
    if len(bad):raise ValueError(f'规则与冻结弱标签不一致：{len(bad)}行。请先检查输入/词典，不继续混合分析。')
    return {'N':len(df),'positive_N':int(labels.sum()),'mismatch_N':0}

def load_scoring_model(name):
    folder=target_folder(name)/'冻结新版模型'
    meta=json.loads((folder/'词表IDF.json').read_text(encoding='utf-8'))
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,3),min_df=2,max_df=1.,norm='l2',preprocessor=training.norm,lowercase=False,sublinear_tf=False,vocabulary=meta['vocabulary'])
    vectorizer.idf_=np.asarray(meta['idf'],float)
    weights=np.load(folder/'L1模型.npz',allow_pickle=False)
    return vectorizer,weights['coef'][0],float(weights['intercept'][0])

def score_target(name,df=None,plan=None,verify=True):
    """从原文重新做TF-IDF transform及模型预测；不是重新训练。"""
    if df is None or plan is None:df,plan=load_target(name)
    ordered=df.set_index('row_id').loc[plan['test_ids']].reset_index()
    vectorizer,coef,intercept=load_scoring_model(name)
    with threadpool_limits(limits=1):
        matrix=vectorizer.transform(ordered.text.tolist())
        probability=expit(matrix@coef+intercept)
    out=ordered[['row_id','family_id','stream','weak_label','text']].copy()
    out['score']=probability
    if verify:
        old=pd.read_csv(target_folder(name)/'冻结新版测试集评分.csv',dtype={'row_id':str})
        if old.row_id.tolist()!=out.row_id.tolist():raise ValueError('评分顺序与冻结新版测试集不同')
        error=float(np.max(np.abs(old.probability-probability)))
        if not np.allclose(old.probability,probability,rtol=1e-10,atol=1e-12):raise ValueError(f'冻结新版评分未复现，最大误差{error:g}')
        out.attrs['max_abs_error_vs_frozen']=error
    return out

def train_target(name,output_base=None):
    """实际完整5折CV+全train网格拟合；原冻结分组，不重新分组/调seed。"""
    target=TARGETS[name][0];df,plan=load_target(name)
    check_weak_labels(name,df)
    lookup={r.row_id:{'row_id':r.row_id,'family_id':r.family_id,'stream':r.stream,'text':r.text,'labels_context':{target:int(r.weak_label)},'labels_surface':{target:int(r.surface_label)}}for r in df.itertuples(index=False)}
    families=dict(zip(df.row_id,df.family_id))
    if output_base is None:
        output_base=target_folder(name)/('重跑模型_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output_base=Path(output_base)
    result=training.fit_route(lookup,plan,families,'lasso_context',target,output_base,lasso_solver)
    actual=output_base/'lasso_context'/target
    old=pd.read_csv(target_folder(name)/'冻结新版测试集评分.csv')
    new=pd.read_csv(actual/'test_predictions.csv')
    old_selection=json.loads((target_folder(name)/'冻结新版模型/模型选择.json').read_text())
    new_selection=json.loads((actual/'selection.json').read_text())
    if new.row_id.tolist()!=old.row_id.tolist():raise ValueError('真实重训测试ID不一致')
    if new_selection['lambda_1se']!=old_selection['lambda_1se']:raise ValueError('重训lambda_1se与冻结新版结果不同；请检查环境')
    error=float(np.max(np.abs(new.probability-old.probability)))
    if not np.allclose(new.probability,old.probability,rtol=1e-7,atol=1e-8):
        warnings.warn(f'完整重训完成，但与冻结新版评分有差异，最大绝对差={error:g}。请勿覆盖冻结新版关联数据。')
    return {'target':name,'fits':result['fits'],'lambda_1se':new_selection['lambda_1se'],'max_score_abs_error_vs_frozen':error,'output':str(actual.relative_to(ROOT))if actual.is_relative_to(ROOT)else str(actual)}

def analyze_target(name,scores=None,x_rows=None):
    """从评分与X重新join到冻结测试侧，再运行99,999次WCR11。"""
    folder=DATA/TARGETS[name][2]
    frozen=pd.read_csv(folder/'关联分析.csv',dtype={'row_id':str},keep_default_na=False)
    if scores is None:scores=score_target(name)
    if x_rows is None:x_rows=compute_x()[0]
    keep=x_rows[x_rows.X.notna()][['row_id','X','ENV']]
    merged=scores[['row_id','score']].merge(keep,on='row_id',how='inner',validate='one_to_one')
    if merged.row_id.tolist()!=frozen.row_id.tolist():raise ValueError('X交集不等于冻结关联样本，不允许默默替换')
    if not np.allclose(merged.X,frozen.X.astype(float),rtol=0,atol=1e-14):raise ValueError('X与冻结关联表不一致')
    if merged.ENV.tolist()!=frozen.ENV.tolist():raise ValueError('ENV与冻结关联表不一致')
    if not np.allclose(merged.score,frozen.score.astype(float),rtol=1e-10,atol=1e-12):raise ValueError('score与冻结关联表不一致')
    cfg=json.loads((ROOT/'python/关联检验配置.json').read_text(encoding='utf-8'))
    with threadpool_limits(limits=1):
        result=cluster_inference_p._analyze(merged.score.to_numpy(),merged.X.to_numpy(),merged.ENV.tolist(),cfg)
    if result['status']!='available':raise ValueError('聚类检验不可用：'+str(result['reasons']))
    return merged,result

def compute_x():
    """从已核对规则事实、coverage与冻结匹配元数据重新计算X。"""
    import x_scoring
    if Path(x_scoring.__file__).resolve().parent != ROOT / "python":
        raise RuntimeError("当前内核的X模块来自另一个项目版本，请重启内核")
    return x_scoring.compute_x()
