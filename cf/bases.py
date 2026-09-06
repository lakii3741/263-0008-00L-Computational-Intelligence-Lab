"""Base member library and the train_bases() helper.

Each lift script in scripts/ calls train_bases(data, level=...) to get the
out-of-fold + test predictions of the requested base library, then runs its
specific stacker on top. There are no cached files on disk -- each script
trains its bases from raw data.

Levels:
    'small' = 14 classical members (used by linear_ensemble.py)
    'mid'   = 39 members  = small + 9 SVDpp variants + 9 lr-decay variants + 7 bag/EASE-binary
              (used by gbm_stacking, gbm_stacking_with_features, gbm_tuned, hyperparam_sweep)
    'full'  = 47 members  = mid + 8 nowish/SVDImpute/bias members
              (used by full_pipeline)
"""

import time
import numpy as np

from cf.cv import oof_predictions
from cf.svdpp import SVDpp
from cf.knn import ItemKNN
from cf.ease import EASE
from cf.ease_binary import EASEBinary
from cf.bag import Bagged
from cf.baselines import IterSVD, BiasModel, SVDImpute
from cf.normalize import ScientistMean, PaperMean
from cf.wrap import Normalized

from cil.predictors import ItemPopPredictor, BPRPredictor, IALSPredictor, EASEBinaryPredictor



def get_members(level):
    """Return the list of (name, factory) for the requested level."""
    if level == "small":
        return _small_members()
    if level == "mid":
        return _small_members() + _mid_extra_members()
    if level == "full":
        return _small_members() + _mid_extra_members() + _full_extra_members()
    # Ablation: number instances 
    if level == "num_base_small":
        return _small_number_base_predictors()
    if level == "num_base_mid":
        return _medium_number_base_predictors()
    # Ablation: wishlist
    if level == "no_wishlist":
        return _small_members(use_wishlist=False) + _mid_extra_members(use_wishlist=False) + _full_extra_members(use_wishlist=False)
    if level == "implicit_wishlist":
        return get_members("full")
    if level == "implicit_and_explicit_wishlist":
        return _disjoint_wishlist_members() + _small_members() + _mid_extra_members() + _full_extra_members()
    # Ablation: components
    if level == "no_svdpp":
        return _small_members(use_svdpp=False) + _mid_extra_members(use_svdpp=False) + _full_extra_members(use_svdpp=False)
    if level == "no_knn":
        return _small_members(use_knn=False) + _mid_extra_members(use_knn=False) + _full_extra_members(use_knn=False)
    if level == "no_bagged_svdpp":
        return _small_members(use_bagged_svdpp=False) + _mid_extra_members(use_bagged_svdpp=False) + _full_extra_members(use_bagged_svdpp=False)
    if level == "no_ease":
        return _small_members(use_ease=False) + _mid_extra_members(use_ease=False) + _full_extra_members(use_ease=False)
    if level == "no_cold_start":
        return _small_members(use_cold_start=False) + _mid_extra_members(use_cold_start=False) + _full_extra_members(use_cold_start=False)
    
# ===========================================
# Ablation: number instances
# ===========================================

def _small_number_base_predictors():
    """1 instance of each of the 14 base predictor groups."""
    members = []
    add = lambda n, f: members.append((n, f))
    add("svdpp_k25_r05_s42",      lambda: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=42))
    add("svdpp_k25_lrdecay_s42",  lambda: SVDpp(k=25, n_epochs=40, lr=0.007, reg=0.05, seed=42, lr_decay=0.95))
    add("svdpp_k25_nowish_s42",   lambda: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=42, include_wishlist=False))
    add("svdpp_k25_r03_long_s42", lambda: SVDpp(k=25, n_epochs=50, lr=0.005, reg=0.03, seed=42, lr_decay=0.97))
    add("svdpp_k25_sci_s42",      lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=42), ScientistMean()))
    add("svdpp_k25_pap_s42",      lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.07, seed=42), PaperMean()))
    add("svdpp_k25_nowish_sci_s42", lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=42, include_wishlist=False), ScientistMean()))
    add("knn_k50_s100",           lambda: ItemKNN(k=50, shrinkage=100))
    add("ease_l150",              lambda: EASE(lam=150))
    add("ease_bin_l200",          lambda: EASEBinary(lam=200, use_wishlist=True))
    add("itersvd_k10",            lambda: IterSVD(k=10, n_iters=20))
    add("bag_svdpp_k25_f7_s101",  lambda: Bagged(lambda: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=101), frac=0.7, seed=101))
    add("svdimpute_k3",           lambda: SVDImpute(k=3))
    add("bias_only",              lambda: BiasModel(shrink=25))

    return members

def _medium_number_base_predictors():
    """~30 members: small + 2 more instances of each group."""
    members = _small_number_base_predictors()
    add = lambda n, f: members.append((n, f))

    add("svdpp_k25_r05_s123",     lambda: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=123))
    add("svdpp_k30_r07_s42",      lambda: SVDpp(k=30, n_epochs=30, lr=0.005, reg=0.07, seed=42))
   
    add("svdpp_k25_lrdecay_s123", lambda: SVDpp(k=25, n_epochs=40, lr=0.007, reg=0.05, seed=123, lr_decay=0.95))
    add("svdpp_k25_lrdecay_s7",   lambda: SVDpp(k=25, n_epochs=40, lr=0.007, reg=0.05, seed=7,   lr_decay=0.95))
  
    add("svdpp_k25_nowish_s123",  lambda: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=123, include_wishlist=False))

    add("svdpp_k25_sci_s7",       lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=7), ScientistMean()))
    
    add("svdpp_k35_r07_pap",      lambda: Normalized(SVDpp(k=35, n_epochs=30, lr=0.005, reg=0.07, seed=42), PaperMean()))
    
    add("knn_k30_s80",            lambda: ItemKNN(k=30, shrinkage=80))
    add("knn_k20_s50",            lambda: ItemKNN(k=20, shrinkage=50))
    
    add("ease_l400",              lambda: EASE(lam=400))
   
    add("ease_bin_l500",          lambda: EASEBinary(lam=500, use_wishlist=True))
   
    add("itersvd_k15",            lambda: IterSVD(k=15, n_iters=20))
    
    add("bag_svdpp_k25_f7_s202",  lambda: Bagged(lambda: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=202), frac=0.7, seed=202))
   
    add("svdimpute_k5",           lambda: SVDImpute(k=5))
    return members 

# ========================================================================
# Ablation: Explicitly adding disjoint wishlist signal with ranking models
# ========================================================================
def _disjoint_wishlist_members():
    members = []
    add = lambda n, f: members.append((n, f))
    add("bpr_32", lambda: BPRPredictor(n_factors=32))
    add("bpr_64", lambda: BPRPredictor(n_factors=64))
    add("pop",    lambda: ItemPopPredictor())
    add("ials_32", lambda: IALSPredictor(n_factors=32))
    add("ials_64", lambda: IALSPredictor(n_factors=64))
    add("ease_bin_l150", lambda: EASEBinaryPredictor(lam=150, use_wishlist=True))
    return members

# ===========================================
# Other
# ===========================================

def _small_members(use_wishlist=True, use_svdpp = True, use_knn=True, use_ease=True, use_bagged_svdpp=True, use_cold_start=True):
    """14 classical CF base members (the 'small' library)."""
    members = []
    add = lambda n, f: members.append((n, f))

    # 4 SVD++ k=25 r=0.05 with different seeds
    if use_svdpp:
        for seed in (42, 123, 7, 31):
            add(f"svdpp_k25_r05_s{seed}",
                lambda s=seed: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=s, include_wishlist=use_wishlist))
        # 2 SVD++ k=30 r=0.07 with different seeds
        for seed in (42, 123):
            add(f"svdpp_k30_r07_s{seed}",
                lambda s=seed: SVDpp(k=30, n_epochs=30, lr=0.005, reg=0.07, seed=s, include_wishlist=use_wishlist))
        # SVD++ wrapped in scientist-mean / paper-mean normalisers
        add("svdpp_k25_r05_sci",
            lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=42, include_wishlist=use_wishlist), ScientistMean()))
        add("svdpp_k25_r05_pap",
            lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=42, include_wishlist=use_wishlist), PaperMean()))
    
    # 3 ItemKNN at different (k, shrinkage)
    if use_knn:
        add("knn_k50_s100", lambda: ItemKNN(k=50, shrinkage=100))
        add("knn_k30_s80",  lambda: ItemKNN(k=30, shrinkage=80))
        add("knn_k20_s50",  lambda: ItemKNN(k=20, shrinkage=50))
    
    # 2 EASE at different lambdas
    if use_ease:
        add("ease_l150", lambda: EASE(lam=150))
        add("ease_l400", lambda: EASE(lam=400))
    
    # IterSVD
    if use_cold_start:
        add("itersvd_k10", lambda: IterSVD(k=10, n_iters=20))
    return members


def _mid_extra_members(use_wishlist=True, use_svdpp = True, use_knn=True, use_ease=True, use_bagged_svdpp=True, use_cold_start=True):
    """25 additional members on top of the small library, taking it to 39."""
    members = []
    add = lambda n, f: members.append((n, f))

    # 3 more SVD++ k=25 r=0.05 seeds
    if use_svdpp:
        for seed in (11, 99, 5):
            add(f"svdpp_k25_r05_s{seed}",
                lambda s=seed: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=s, include_wishlist=use_wishlist))
        # SVDpp at different ranks / regs
        add("svdpp_k20_r04_s42",
            lambda: SVDpp(k=20, n_epochs=30, lr=0.005, reg=0.04, seed=42, include_wishlist=use_wishlist))
        add("svdpp_k35_r06_s42",
            lambda: SVDpp(k=35, n_epochs=30, lr=0.005, reg=0.06, seed=42, include_wishlist=use_wishlist))
        add("svdpp_k25_r05_sci_s7",
            lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=7, include_wishlist=use_wishlist), ScientistMean()))
    
    # smaller and larger KNN
    if use_knn:
        add("knn_k10_s30",  lambda: ItemKNN(k=10, shrinkage=30))
        add("knn_k60_s120", lambda: ItemKNN(k=60, shrinkage=120))
    
    # extra IterSVD rank
    if use_cold_start:
        add("itersvd_k15", lambda: IterSVD(k=15, n_iters=20))

    # 5 SVD++ with lr-decay (different convergence path)
    if use_svdpp:
        for seed in (42, 123, 7, 31, 11):
            add(f"svdpp_k25_lrdecay_s{seed}",
                lambda s=seed: SVDpp(k=25, n_epochs=40, lr=0.007, reg=0.05, seed=s, lr_decay=0.95, include_wishlist=use_wishlist))
        # higher-capacity SVDpp
        add("svdpp_k50_r08_s42",
            lambda: SVDpp(k=50, n_epochs=30, lr=0.005, reg=0.08, seed=42, include_wishlist=use_wishlist))
        add("svdpp_k50_r08_s7",
            lambda: SVDpp(k=50, n_epochs=30, lr=0.005, reg=0.08, seed=7, include_wishlist=use_wishlist))
        # less reg, longer training
        add("svdpp_k25_r03_long_s42",
            lambda: SVDpp(k=25, n_epochs=50, lr=0.005, reg=0.03, seed=42, lr_decay=0.97, include_wishlist=use_wishlist))
        # paper-norm at higher k
        add("svdpp_k35_r07_pap",
            lambda: Normalized(SVDpp(k=35, n_epochs=30, lr=0.005, reg=0.07, seed=42, include_wishlist=use_wishlist), PaperMean()))

    # 4 bagged SVDpp (bootstrap-sampled training data)
    if use_bagged_svdpp:
        for seed, frac in [(101, 0.7), (202, 0.7), (303, 0.8), (404, 0.6)]:
            add(f"bag_svdpp_k25_f{int(frac*10)}_s{seed}",
                lambda s=seed, fr=frac: Bagged(
                    lambda ss=s: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05, seed=ss, include_wishlist=use_wishlist),
                    frac=fr, seed=s))
    # 3 EASE on the binary R ∪ W co-occurrence matrix
    if use_ease:
        add("ease_bin_l200", lambda: EASEBinary(lam=200, use_wishlist=use_wishlist))
        add("ease_bin_l500", lambda: EASEBinary(lam=500, use_wishlist=use_wishlist))
        add("ease_bin_l50",  lambda: EASEBinary(lam=50,  use_wishlist=use_wishlist))
    return members


def _full_extra_members(use_wishlist=True, use_svdpp = True, use_knn=True, use_ease=True, use_bagged_svdpp=True, use_cold_start=True):
    """8 additional class-diverse members on top of mid, taking it to 47."""
    members = []
    add = lambda n, f: members.append((n, f))

    # 3 SVD++ with rating-only Y-aggregator (no wishlist)
    if  use_svdpp:
        for seed in (42, 123, 7):
            add(f"svdpp_k25_nowish_s{seed}",
                lambda s=seed: SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05,
                                     seed=s, include_wishlist=False))
        # smaller-rank SVDpp
        add("svdpp_k15_r04_s42",
            lambda: SVDpp(k=15, n_epochs=30, lr=0.005, reg=0.04, seed=42, include_wishlist=use_wishlist))
        # nowish SVDpp wrapped in scientist-mean
        add("svdpp_k25_nowish_sci",
            lambda: Normalized(SVDpp(k=25, n_epochs=30, lr=0.005, reg=0.05,
                                     seed=42, include_wishlist=False),
                               ScientistMean()))
        
    # very-small-rank SVDImpute and a pure bias model for very-cold-start diversity
    if use_cold_start:
        add("svdimpute_k3", lambda: SVDImpute(k=3))
        add("svdimpute_k5", lambda: SVDImpute(k=5))
        add("bias_only", lambda: BiasModel(shrink=25))
    return members


def train_bases(data, level="mid", verbose=True):
    """Trains all base members of the requested level via 5-fold CV.

    Returns (Z, P, y, names) where:
        Z[i, j] = OOF prediction of member j for training row i
        P[i, j] = test-set prediction of member j refit on the full train data
        y[i]   = true rating of training row i
        names  = list of member names (length M)
    """
    members = get_members(level)
    M = len(members)
    N = len(data.ratings)
    T = len(data.test_sids)
    Z = np.zeros((N, M))
    P = np.zeros((T, M))
    y = data.ratings["rating"].to_numpy(dtype=np.float64)

    for j, (name, factory) in enumerate(members):
        if verbose:
            print(f"[{j+1}/{M}] {name}", flush=True)
        t0 = time.time()
        Z[:, j], _ = oof_predictions(
            data,
            lambda tr, w, f=factory: f().fit(tr, data.n_users, data.n_items, w),
            n_folds=5, seed=42, verbose=False,
        )
        pred = factory().fit(data.ratings, data.n_users, data.n_items, data.wishlist)
        P[:, j] = np.clip(pred.predict(data.test_sids, data.test_pids), 1.0, 5.0)
        if verbose:
            print(f"    done ({time.time()-t0:.0f}s)", flush=True)

    return Z, P, y, [n for n, _ in members]


def row_features(data, sids, pids, include_wishlist=True):
    """Build the 9 per-(user, item) context features used by the stacker:
        log(1 + n_user_ratings), log(1 + n_item_ratings),
        shrunk user mean rating, shrunk item mean rating,
        user rating std, item rating std,
        log(1 + n_user_wishlist), log(1 + n_item_wishlist),
        has_wish (1 if (sid, pid) appears in train_tbr).

    If include_wishlist is false, the final three wishlist-derived features are
    zeroed so ablations can remove every wishlist signal path.
    """
    rs = data.ratings["sid"].to_numpy()
    rp = data.ratings["pid"].to_numpy()
    r  = data.ratings["rating"].to_numpy(dtype=np.float64)
    mu = r.mean()

    u_n = np.bincount(rs, minlength=data.n_users).astype(np.float64)
    i_n = np.bincount(rp, minlength=data.n_items).astype(np.float64)
    K = 25.0
    u_sum = np.zeros(data.n_users); np.add.at(u_sum, rs, r)
    i_sum = np.zeros(data.n_items); np.add.at(i_sum, rp, r)
    u_mean = (u_sum + K * mu) / (u_n + K)
    i_mean = (i_sum + K * mu) / (i_n + K)

    u_sq = np.zeros(data.n_users); np.add.at(u_sq, rs, r ** 2)
    i_sq = np.zeros(data.n_items); np.add.at(i_sq, rp, r ** 2)
    u_var = np.maximum(u_sq / np.maximum(u_n, 1) - (u_sum / np.maximum(u_n, 1)) ** 2, 0)
    i_var = np.maximum(i_sq / np.maximum(i_n, 1) - (i_sum / np.maximum(i_n, 1)) ** 2, 0)
    u_std = np.where(u_n >= 2, np.sqrt(u_var), 1.0)
    i_std = np.where(i_n >= 2, np.sqrt(i_var), 1.0)

    if include_wishlist:
        ws = data.wishlist["sid"].to_numpy()
        wp = data.wishlist["pid"].to_numpy()
        u_w = np.bincount(ws, minlength=data.n_users).astype(np.float64)
        i_w = np.bincount(wp, minlength=data.n_items).astype(np.float64)
        wishset = set(zip(ws.tolist(), wp.tolist()))
        has_wish = np.array([(int(s), int(p)) in wishset for s, p in zip(sids, pids)],
                            dtype=np.float32)
    else:
        u_w = np.zeros(data.n_users, dtype=np.float64)
        i_w = np.zeros(data.n_items, dtype=np.float64)
        has_wish = np.zeros(len(sids), dtype=np.float32)

    return np.stack([
        np.log1p(u_n[sids]), np.log1p(i_n[pids]),
        u_mean[sids], i_mean[pids],
        u_std[sids], i_std[pids],
        np.log1p(u_w[sids]), np.log1p(i_w[pids]),
        has_wish,
    ], axis=1).astype(np.float32)
