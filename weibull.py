import numpy as np
from scipy.special import gamma
from config import BASE_PARAMS


ENV_FACTOR = {
    "manzana": {"nevera": 1.5, "ambiente": 1.0},
    "banano": {"nevera": 1.4, "ambiente": 1.0},
    "tomate": {"nevera": 1.2, "ambiente": 1.0},
    "zanahoria": {"nevera": 1.7, "ambiente": 1.0},
    "pepino": {"nevera": 1.4, "ambiente": 1.0},
    "mango": {"nevera": 1.1, "ambiente": 1.0},
    "naranja": {"nevera": 1.5, "ambiente": 1.0},
    "papa": {"nevera": 1.2, "ambiente": 1.0},
    "pimenton": {"nevera": 1.6, "ambiente": 1.0}
}


PARAMS = {
    "manzana": {"w_s": 0.25, "w_m": 0.8, "w_w": 0.3, "w_c": 0.2, "w_g": 0.25, "min_factor": 0.55},
    "banano": {"w_s": 0.05, "w_m": 0.4, "w_w": 0.35, "w_c": 0.6, "w_g": 0.15, "min_factor": 0.65},  
    "tomate": {"w_s": 0.2, "w_m": 1.1, "w_w": 0.25, "w_c": 0.3, "w_g": 0.3, "min_factor": 0.55}, 
    "zanahoria": {"w_s": 0.15, "w_m": 0.5, "w_w": 0.8, "w_c": 0.15, "w_g": 0.25, "min_factor": 0.6}, 
    "pepino": {"w_s": 0.2, "w_m": 0.6, "w_w": 0.7, "w_c": 0.2, "w_g": 0.25, "min_factor": 0.55},
    "mango": {"w_s": 0.2, "w_m": 0.5, "w_w": 0.6, "w_c": 0.9, "w_g": 0.3, "min_factor": 0.5}, 
    "naranja": {"w_s": 0.3, "w_m": 0.6, "w_w": 0.15, "w_c": 0.15, "w_g": 0.25, "min_factor": 0.65},
    "papa": {"w_s": 0.8, "w_m": 0.4, "w_w": 0.15, "w_c": 0.1, "w_g": 0.25, "min_factor": 0.7},  
    "pimenton": {"w_s": 0.25, "w_m": 0.7, "w_w": 0.6, "w_c": 0.4, "w_g": 0.25, "min_factor": 0.5}
}

GLOBAL_MAP = {0:0.0,1:0.2,2:0.45,3:0.8}


def predict_days(product,s,m,w,c,g,env="ambiente"):

    lam,k = BASE_PARAMS[product]
    p = PARAMS[product]

    g_score = GLOBAL_MAP.get(g,0.45)
    env_factor = ENV_FACTOR.get(product,{}).get(env,1.0)

    D = (
        p["w_s"]*s +
        p["w_m"]*m +
        p["w_w"]*w +
        p["w_c"]*c +
        p["w_g"]*g_score
    )

    lam_env = lam * env_factor

    lam_adj = lam_env * np.exp(-D)

    lam_min = p["min_factor"] * lam_env
    lam_adj = max(lam_adj, lam_min)

    return lam_adj * gamma(1 + 1/k)