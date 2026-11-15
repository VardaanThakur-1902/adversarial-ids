"""Phase 5: feature groups, scaling helpers, repair() and check() for IoTID20 adversarial work.
Threat model v1: the attacker can PAD payloads (more bytes) and DELAY packets (longer times).
Packet counts, ports, protocol, flags, headers and minima are NOT changed; values only go UP."""
import numpy as np, pandas as pd

INC_SIZE = ["TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts", "Fwd_Pkt_Len_Max", "Bwd_Pkt_Len_Max"]
INC_TIME = ["Flow_Duration", "Fwd_IAT_Tot", "Bwd_IAT_Tot", "Flow_IAT_Max", "Fwd_IAT_Max", "Bwd_IAT_Max"]
FREE_STD = ["Fwd_Pkt_Len_Std", "Bwd_Pkt_Len_Std"]
INCREASE_ONLY = INC_SIZE + INC_TIME
CONTROLLABLE = INCREASE_ONLY + FREE_STD
DERIVED = ["Subflow_Fwd_Byts", "Subflow_Bwd_Byts", "Fwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Mean",
           "Fwd_Seg_Size_Avg", "Bwd_Seg_Size_Avg", "Pkt_Len_Max", "Pkt_Len_Var", "Pkt_Len_Std",
           "Pkt_Len_Mean", "Pkt_Size_Avg", "Flow_Byts/s", "Flow_Pkts/s", "Fwd_Pkts/s", "Bwd_Pkts/s",
           "Flow_IAT_Mean", "Fwd_IAT_Mean", "Bwd_IAT_Mean", "Flow_IAT_Std", "Fwd_IAT_Std", "Bwd_IAT_Std"]

def groups(features):
    fixed = [f for f in features if f not in CONTROLLABLE and f not in DERIVED]
    return {"controllable": [f for f in features if f in CONTROLLABLE],
            "derived": [f for f in features if f in DERIVED], "fixed": fixed}

def controllable_mask(features):
    return np.array([f in CONTROLLABLE for f in features])

# ---------- scaling helpers (raw <-> scaled) ----------
def to_scaled(Xraw, scaler, log_cols, features):
    D = Xraw[features].copy()
    D[log_cols] = np.log1p(D[log_cols].clip(lower=0))
    return np.clip(scaler.transform(D), 0, 1)

def to_raw(Xs, scaler, log_cols, features):
    D = pd.DataFrame(scaler.inverse_transform(Xs), columns=features)
    D[log_cols] = np.expm1(D[log_cols])
    return D

# ---------- helpers ----------
def _std_bound(mx, mn, mean, m):
    """Bhatia-Davis upper bound on the sample std of m values lying in [mn, mx] with the given mean."""
    return np.sqrt(np.clip((mx - mean) * (mean - mn), 0, None) * m / np.maximum(m - 1, 1))

def _comb_std(tf, tb, mf, mb, sf, sb):
    n = tf + tb; M = np.where(n > 0, (tf * mf + tb * mb) / np.maximum(n, 1), 0)
    v = (np.clip(tf - 1, 0, None) * sf**2 + np.clip(tb - 1, 0, None) * sb**2
         + tf * (mf - M)**2 + tb * (mb - M)**2) / np.maximum(n - 1, 1)
    return np.sqrt(np.clip(v, 0, None))

def repair(Xn, Xo):
    """Make a perturbed raw sample set Xn consistent. Xo = the original raw samples (same order/index)."""
    X = Xn.copy().reset_index(drop=True); Xo = Xo.reset_index(drop=True)
    tf, tb = Xo["Tot_Fwd_Pkts"].values, Xo["Tot_Bwd_Pkts"].values; n = tf + tb
    g = groups(list(Xo.columns))
    for c in g["fixed"]: X[c] = Xo[c]                                   # 1. fixed features never change
    for c in INCREASE_ONLY: X[c] = np.maximum(np.round(X[c]), Xo[c])    # 2. integers, increase only
    for c in FREE_STD: X[c] = np.maximum(X[c], 0)
    # 3. a direction with no packets cannot change; IATs need >=2 packets
    # (a single packet has min == max == total, so a direction with <2 packets cannot be padded)
    for m, cols in [(tf <= 1, ["TotLen_Fwd_Pkts", "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Std"]),
                    (tb <= 1, ["TotLen_Bwd_Pkts", "Bwd_Pkt_Len_Max", "Bwd_Pkt_Len_Std"]),
                    (tf <= 1, ["Fwd_IAT_Tot", "Fwd_IAT_Max"]), (tb <= 1, ["Bwd_IAT_Tot", "Bwd_IAT_Max"])]:
        for c in cols: X.loc[m, c] = Xo.loc[m, c]
    # 4. sizes: with min fixed, total bytes must lie in [max + (k-1)*min, min + (k-1)*max]
    for p, cnt in [("Fwd", tf), ("Bwd", tb)]:
        L, Mx, Mn = X[f"TotLen_{p}_Pkts"].values, X[f"{p}_Pkt_Len_Max"].values, Xo[f"{p}_Pkt_Len_Min"].values
        lo, hi = Mx + (cnt - 1) * Mn, Mn + (cnt - 1) * Mx
        L2 = np.maximum(np.minimum(np.maximum(L, lo), hi), Xo[f"TotLen_{p}_Pkts"].values)
        X[f"TotLen_{p}_Pkts"] = np.where(cnt > 1, L2, Xo[f"TotLen_{p}_Pkts"].values)
    # 5. timing: max IAT between its mean and its total; duration covers everything
    for p, cnt in [("Fwd", tf), ("Bwd", tb)]:
        act = cnt > 1; tot = X[f"{p}_IAT_Tot"].values
        mean = np.where(act, tot / np.maximum(cnt - 1, 1), 0)
        mx = np.minimum(np.maximum(X[f"{p}_IAT_Max"].values, np.ceil(mean)), tot)
        X[f"{p}_IAT_Max"] = np.where(act, mx, Xo[f"{p}_IAT_Max"].values)
    dur = np.max(np.vstack([X["Flow_Duration"], X["Fwd_IAT_Tot"], X["Bwd_IAT_Tot"], X["Flow_IAT_Max"],
                            Xo["Idle_Max"], Xo["Active_Max"]]), axis=0)
    X["Flow_Duration"] = dur
    fmean = np.where(n > 1, dur / np.maximum(n - 1, 1), Xo["Flow_IAT_Mean"].values)
    X["Flow_IAT_Max"] = np.minimum(np.maximum(X["Flow_IAT_Max"], np.ceil(fmean)), dur)
    # 6. derived features recomputed from the controllable ones
    Lf, Lb = X["TotLen_Fwd_Pkts"].values, X["TotLen_Bwd_Pkts"].values
    fm = np.where(tf > 0, Lf / np.maximum(tf, 1), 0); bm = np.where(tb > 0, Lb / np.maximum(tb, 1), 0)
    X["Subflow_Fwd_Byts"], X["Subflow_Bwd_Byts"] = Lf, Lb
    X["Fwd_Pkt_Len_Mean"] = X["Fwd_Seg_Size_Avg"] = fm
    X["Bwd_Pkt_Len_Mean"] = X["Bwd_Seg_Size_Avg"] = bm
    X["Pkt_Len_Max"] = np.maximum(X["Fwd_Pkt_Len_Max"], X["Bwd_Pkt_Len_Max"])
    # std of packet length must stay within its feasible range
    for p, cnt, mean in [("Fwd", tf, fm), ("Bwd", tb, bm)]:
        b = _std_bound(X[f"{p}_Pkt_Len_Max"].values, Xo[f"{p}_Pkt_Len_Min"].values, mean, cnt)
        X[f"{p}_Pkt_Len_Std"] = np.where(cnt > 1, np.minimum(X[f"{p}_Pkt_Len_Std"].values, b),
                                         Xo[f"{p}_Pkt_Len_Std"].values)
    M_old = (Xo["TotLen_Fwd_Pkts"].values + Xo["TotLen_Bwd_Pkts"].values) / np.maximum(n, 1)
    M_new = (Lf + Lb) / np.maximum(n, 1)
    X["Pkt_Len_Mean"] = Xo["Pkt_Len_Mean"] + (M_new - M_old)
    X["Pkt_Size_Avg"] = Xo["Pkt_Size_Avg"] + (M_new - M_old) * (n + 1) / np.maximum(n, 1)
    c_old = _comb_std(tf, tb, Xo["Fwd_Pkt_Len_Mean"].values, Xo["Bwd_Pkt_Len_Mean"].values,
                      Xo["Fwd_Pkt_Len_Std"].values, Xo["Bwd_Pkt_Len_Std"].values)
    c_new = _comb_std(tf, tb, fm, bm, X["Fwd_Pkt_Len_Std"].values, X["Bwd_Pkt_Len_Std"].values)
    X["Pkt_Len_Std"] = np.maximum(Xo["Pkt_Len_Std"] + (c_new - c_old), 0)
    X["Pkt_Len_Var"] = X["Pkt_Len_Std"] ** 2
    X["Flow_Byts/s"] = (Lf + Lb) / dur * 1e6; X["Flow_Pkts/s"] = n / dur * 1e6
    X["Fwd_Pkts/s"] = tf / dur * 1e6;         X["Bwd_Pkts/s"] = tb / dur * 1e6
    X["Flow_IAT_Mean"] = fmean
    for p, cnt in [("Fwd", tf), ("Bwd", tb)]:
        X[f"{p}_IAT_Mean"] = np.where(cnt > 1, X[f"{p}_IAT_Tot"] / np.maximum(cnt - 1, 1), Xo[f"{p}_IAT_Mean"].values)
    # IAT std bounded by its feasible range
    for p, mx, mn, mean, m in [("Flow", X["Flow_IAT_Max"].values, Xo["Flow_IAT_Min"].values, fmean, n - 1),
                               ("Fwd", X["Fwd_IAT_Max"].values, Xo["Fwd_IAT_Min"].values, X["Fwd_IAT_Mean"].values, tf - 1),
                               ("Bwd", X["Bwd_IAT_Max"].values, Xo["Bwd_IAT_Min"].values, X["Bwd_IAT_Mean"].values, tb - 1)]:
        b = _std_bound(mx, mn, mean, m)
        X[f"{p}_IAT_Std"] = np.where(m > 1, np.minimum(Xo[f"{p}_IAT_Std"].values, b), Xo[f"{p}_IAT_Std"].values)
    return X[list(Xo.columns)]

def project(Xs_adv, Xraw_orig, scaler, log_cols, features):
    """scaled adversarial samples -> raw -> repaired -> scaled (what the model should be shown)."""
    raw = repair(to_raw(Xs_adv, scaler, log_cols, features), Xraw_orig)
    return to_scaled(raw, scaler, log_cols, features), raw

# ---------- validity checker ----------
INT_FEATURES = None
def check(X, Xo=None, tol=1e-3):
    """Return a table: rule, rows checked, violation rate. X, Xo are raw DataFrames."""
    X = X.reset_index(drop=True); R = []
    tf, tb = X["Tot_Fwd_Pkts"], X["Tot_Bwd_Pkts"]; n = tf + tb; d = X["Flow_Duration"]
    def add(name, ok, mask=None):
        ok = np.asarray(ok); m = np.ones(len(X), bool) if mask is None else np.asarray(mask)
        R.append((name, int(m.sum()), float(1 - ok[m].mean()) if m.sum() else 0.0))
    close = lambda a, b: np.isclose(a, b, rtol=tol, atol=1e-6)
    neg = [c for c in X.columns if c != "Init_Bwd_Win_Byts"]
    add("no negative values (except Init_Bwd_Win_Byts)", (X[neg] >= -1e-9).all(axis=1))
    ints = ["Flow_Duration","Tot_Fwd_Pkts","Tot_Bwd_Pkts","TotLen_Fwd_Pkts","TotLen_Bwd_Pkts","Fwd_Pkt_Len_Max","Bwd_Pkt_Len_Max",
            "Flow_IAT_Max","Fwd_IAT_Tot","Bwd_IAT_Tot","Fwd_IAT_Max","Bwd_IAT_Max"]
    add("integer-valued size/time features", (X[ints] == X[ints].round()).all(axis=1))
    add("Subflow bytes == total bytes", close(X["Subflow_Fwd_Byts"], X["TotLen_Fwd_Pkts"]) & close(X["Subflow_Bwd_Byts"], X["TotLen_Bwd_Pkts"]))
    add("Seg_Size_Avg == packet-length mean", close(X["Fwd_Seg_Size_Avg"], X["Fwd_Pkt_Len_Mean"]) & close(X["Bwd_Seg_Size_Avg"], X["Bwd_Pkt_Len_Mean"]))
    add("Fwd mean == bytes/packets", close(X["Fwd_Pkt_Len_Mean"], X["TotLen_Fwd_Pkts"] / tf.replace(0, np.nan)), tf > 0)
    add("Bwd mean == bytes/packets", close(X["Bwd_Pkt_Len_Mean"], X["TotLen_Bwd_Pkts"] / tb.replace(0, np.nan)), tb > 0)
    add("Pkt_Len_Var == Pkt_Len_Std^2", close(X["Pkt_Len_Var"], X["Pkt_Len_Std"] ** 2))
    add("Pkt_Len_Max == max(Fwd, Bwd max)", close(X["Pkt_Len_Max"], np.maximum(X["Fwd_Pkt_Len_Max"], X["Bwd_Pkt_Len_Max"])))
    add("Flow_Pkts/s == packets/duration", close(X["Flow_Pkts/s"], n / d * 1e6))
    add("Flow_Byts/s == bytes/duration", close(X["Flow_Byts/s"], (X["TotLen_Fwd_Pkts"] + X["TotLen_Bwd_Pkts"]) / d * 1e6))
    add("Fwd/Bwd_Pkts/s == packets/duration", close(X["Fwd_Pkts/s"], tf / d * 1e6) & close(X["Bwd_Pkts/s"], tb / d * 1e6))
    add("Flow_IAT_Mean == duration/(n-1)", close(X["Flow_IAT_Mean"], d / (n - 1).replace(0, np.nan)), n > 1)
    add("Fwd_IAT_Mean == Fwd_IAT_Tot/(tf-1)", close(X["Fwd_IAT_Mean"], X["Fwd_IAT_Tot"] / (tf - 1).replace(0, np.nan)), tf > 1)
    add("Bwd_IAT_Mean == Bwd_IAT_Tot/(tb-1)", close(X["Bwd_IAT_Mean"], X["Bwd_IAT_Tot"] / (tb - 1).replace(0, np.nan)), tb > 1)
    for p in ["Flow", "Fwd", "Bwd"]:
        add(f"{p}_IAT: min <= mean <= max", (X[f"{p}_IAT_Min"] <= X[f"{p}_IAT_Mean"] + 1e-6) & (X[f"{p}_IAT_Mean"] <= X[f"{p}_IAT_Max"] + 1e-6))
    for p in ["Fwd", "Bwd"]:
        add(f"{p}_Pkt_Len: min <= mean <= max", (X[f"{p}_Pkt_Len_Min"] <= X[f"{p}_Pkt_Len_Mean"] + 1e-6) & (X[f"{p}_Pkt_Len_Mean"] <= X[f"{p}_Pkt_Len_Max"] + 1e-6))
    for p, cnt in [("Fwd", tf), ("Bwd", tb)]:
        L, Mx, Mn = X[f"TotLen_{p}_Pkts"], X[f"{p}_Pkt_Len_Max"], X[f"{p}_Pkt_Len_Min"]
        add(f"{p} bytes within [max+(k-1)*min, min+(k-1)*max]", (L >= Mx + (cnt - 1) * Mn - 1e-6) & (L <= Mn + (cnt - 1) * Mx + 1e-6), cnt > 1)
    add("Fwd_IAT_Max <= Fwd_IAT_Tot, Bwd same", (X["Fwd_IAT_Max"] <= X["Fwd_IAT_Tot"] + 1e-6) & (X["Bwd_IAT_Max"] <= X["Bwd_IAT_Tot"] + 1e-6))
    add("all IATs / idle / active <= duration", (X[["Flow_IAT_Max","Fwd_IAT_Tot","Bwd_IAT_Tot","Idle_Max","Active_Max"]].le(d + 1e-6, axis=0)).all(axis=1))
    add("Fwd_Pkt_Len_Std within feasible bound", X["Fwd_Pkt_Len_Std"] <= _std_bound(X["Fwd_Pkt_Len_Max"], X["Fwd_Pkt_Len_Min"], X["Fwd_Pkt_Len_Mean"], tf) + 1e-6, tf > 1)
    add("Bwd_Pkt_Len_Std within feasible bound", X["Bwd_Pkt_Len_Std"] <= _std_bound(X["Bwd_Pkt_Len_Max"], X["Bwd_Pkt_Len_Min"], X["Bwd_Pkt_Len_Mean"], tb) + 1e-6, tb > 1)
    if Xo is not None:
        Xo = Xo.reset_index(drop=True); g = groups(list(X.columns))
        add("fixed features unchanged", np.isclose(X[g["fixed"]].values, Xo[g["fixed"]].values, rtol=1e-9, atol=1e-9).all(axis=1))
        add("increase-only features did not decrease", (X[INCREASE_ONLY].values >= Xo[INCREASE_ONLY].values - 1e-9).all(axis=1))
    return pd.DataFrame(R, columns=["rule", "rows_checked", "violation_rate"])
