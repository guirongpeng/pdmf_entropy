from __future__ import annotations

import numpy as np

# 全局随机种子等通用配置
DEFAULT_RANDOM_STATE = 3220822


# 均按照原论文的参数网格，无需改动
def get_param_grids(num_features: int) -> dict:
    """
    返回各属性约简算法/新算法的参数网格。
    - FNRS 会在 lamda × alpha 网格上产生一批候选子集；
    - IARFCIE 遍历 threshold；
    - MFIGI: 低维(<=2000特征)用0.002；高维(>2000特征)用0.005
    - MFNMI: theta ∈ [0.1, 0.5] 步长0.05；delta ∈ [0.5, 0.9] 步长0.05
    """
    fnrs_lamdas = [round(x, 2) for x in np.arange(0.1, 0.55, 0.05)]
    fnrs_alphas = [round(x, 2) for x in np.arange(0.5, 1.05, 0.05)]

    iarfcie_thresholds = [round(x, 2) for x in np.arange(0.0, 1.1, 0.1)]

    mfigi_ws = [0.005] if num_features > 2000 else [0.002]
    
    mfnmi_thetas = [round(x, 2) for x in np.arange(0.1, 0.51, 0.05)]
    mfnmi_deltas = [round(x, 2) for x in np.arange(0.5, 0.91, 0.05)]

    return {
        "FNRS": {"lamda": fnrs_lamdas, "alpha": fnrs_alphas},
        "MFIGI": {"w": mfigi_ws},
        "IARFCIE": {"threshold": iarfcie_thresholds},
        "MFNMI": {"theta": mfnmi_thetas, "delta": mfnmi_deltas},
        "MFREN": {"lambda_v": [0.0]},   # 示例：MFREN/FSFrMI 当前固定 lambda_v=0.0，
        "FSFrMI": {"lambda_v": [0.0]},  # 如需扩展可在此添加网格（原论文均值0.0，无需改动）



        # ARPDMF：delta×k×mu_method 笛卡尔积（每折候选数 = |δ|×|k|×|μ|）；与其它算法网格独立
        "ARPDMF": {
            # "delta":[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.009, 0.01],
            "delta":[0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            "k": [1,2, 3, 4, 5, 6, 7, 8, 9, 10],
            "mu_method": ["B"],
        },
    }

# 注册分类模型工厂，无需改动
def get_model_factories(random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """
    统一注册可用的分类模型工厂（尽量使用默认超参，仅设置 random_state / n_jobs / 静默开关）。
    外部依赖（XGBoost/LightGBM/CatBoost）缺失时自动跳过。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    n_jobs = 6
    factories = {
        "RF": lambda: RandomForestClassifier(n_jobs=n_jobs, random_state=random_state),
        "ET": lambda: ExtraTreesClassifier(n_jobs=n_jobs, random_state=random_state),
        "BGG": lambda: BaggingClassifier(random_state=random_state, n_jobs=n_jobs),
        "GB": lambda: GradientBoostingClassifier(random_state=random_state),
        "SVM": lambda: SVC(random_state=random_state),
        "LSVC": lambda: LinearSVC(random_state=random_state),
        "KNN": lambda: KNeighborsClassifier(),
        "GN": lambda: GaussianNB(),
        "LR": lambda: LogisticRegression(n_jobs=n_jobs, random_state=random_state),
        # 供 BGG 作为基学习器时可用，但不直接暴露到分组
        "DT_BASE": lambda: DecisionTreeClassifier(random_state=random_state),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore
        factories["XB"] = lambda: XGBClassifier(n_jobs=n_jobs, random_state=random_state, verbosity=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        factories["LB"] = lambda: LGBMClassifier(n_jobs=n_jobs, random_state=random_state, verbose=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier  # type: ignore
        factories["CB"] = lambda: CatBoostClassifier(random_state=random_state, verbose=False)
    except Exception:
        pass
    return factories

# 六大类分组：可按需增删模型代号
def get_model_groups_raw() -> dict:
    """
    六大类分组（可在此增删模型代号）。
    说明：DT_BASE 仅作为 BGG 的内部基学习器，不列入组。
    """
    return {
        "线性模型": ["LR"],
        "基于实例的方法": ["KNN"],
        "Bagging集成": ["RF", "ET", "BGG"],
        "Boosting集成": ["GB", "XB", "LB", "CB"],
        "支持向量机家族": ["SVM", "LSVC"],
        "概率模型": ["GN"],
    }

# 指定启用的属性约简算法名称列表，可按需增删来控制本次实验参与的算法
def get_enabled_reductions() -> list[str]:
    """
    指定启用的属性约简算法名称列表（与注册/主流程中的算法名一致）。
    可按需增删来控制本次实验参与的算法。
    """
    return [
        # "MFREN",  # 如需启用可取消注释，pass
        # "MFNMI",    # 新算法pgr


        "FNRS",
        "FRAR",
        "MFIGI",
        "FSFrMI",
        "IARFCIE",
        "ARPDMF",   # zc
        "ALL",
        

    ]
