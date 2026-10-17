"""Pure rolling-window drift detection used by the FastAPI monitoring endpoint."""


def drift_signals(window, baseline, conf_drop=0.10, unk_rise=0.12, min_window=20):
    """Compare recent (confidence, unknown-token-rate, is_scam) tuples to baseline.

    Return a dict consistent with /monitor and /predict call sites.
    Insufficient messages produce WARMING_UP rather than a false OK verdict.
    """
    n = len(window)
    if n < min_window:
        return {"status": "WARMING_UP", "n": n, "signals": []}
    if not window:
        raise ValueError("Cannot evaluate an empty window")
    mean_conf = sum(float(row[0]) for row in window) / n
    mean_unk = sum(float(row[1]) for row in window) / n
    scam_rate = sum(bool(row[2]) for row in window) / n
    signals = []
    baseline_conf = float(baseline["mean_conf"])
    baseline_unk = float(baseline["unk_rate"])
    if baseline_conf - mean_conf > conf_drop:
        signals.append(f"confidence dropped {baseline_conf:.2f} -> {mean_conf:.2f}")
    if mean_unk - baseline_unk > unk_rise:
        signals.append(f"unknown-token rate {baseline_unk:.0%} -> {mean_unk:.0%}")
    return {
        "status": "DRIFT" if signals else "OK",
        "signals": signals,
        "n": n,
        "mean_conf": round(mean_conf, 3),
        "unk_rate": round(mean_unk, 3),
        "scam_rate": round(scam_rate, 3),
    }
