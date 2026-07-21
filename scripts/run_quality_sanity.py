"""
Sanity checks for the Quality & Health metrics (quality/dupont.py,
quality/altman.py, quality/piotroski.py, quality/beneish.py).

Same candidate set as scripts/run_dcf_sanity.py for continuity (AAPL, MSFT,
KO, NVDA), plus the same bank/insurer/REIT exclusion spot-checks
(JPM/MET/O) to confirm quality/inputs.py's reuse of dcf/exclusions.py works
end to end.

What "sanity" means here: no single ground truth to check every number
against, but structural plausibility is the bar —
  1. Every ticker runs without error for all 4 metrics.
  2. DuPont's recomposed ROE (product of the 5 components) should be close
     to net_income/equity computed directly — a large gap would indicate a
     wiring bug in one of the components, not a real financial fact.
  3. Altman Z'' zone and Piotroski F-Score / Beneish M-Score land in a
     plausible range for each company's known profile (e.g. AAPL/MSFT/KO
     shouldn't land in Altman's distress zone; a real F-Score of 0 or 9
     across the board for every ticker would suggest a signal-direction bug).
  4. JPM/MET/O are confirmed excluded from Altman/Piotroski/Beneish with the
     expected reason; DuPont still computes for them (with the caveat flag).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quality.dupont import compute_dupont
from quality.altman import compute_altman_z
from quality.piotroski import compute_piotroski_f
from quality.beneish import compute_beneish_m

CANDIDATES = ["AAPL", "MSFT", "KO", "NVDA"]
EXCLUSION_SPOT_CHECKS = [("JPM", "bank"), ("MET", "insurer"), ("O", "reit")]


def _print_dupont(ticker):
    d = compute_dupont(ticker)
    print(f"  DuPont: {d}")
    if d["status"] == "ok":
        gap = abs(d["roe"] - d["roe_direct"])
        flag = " <-- MISMATCH" if gap > 0.01 else ""
        print(f"    roe_recomposed={d['roe']:.4f} vs roe_direct={d['roe_direct']:.4f} (gap={gap:.4f}){flag}")


def _print_altman(ticker):
    a = compute_altman_z(ticker)
    print(f"  Altman Z'': {a}")


def _print_piotroski(ticker):
    p = compute_piotroski_f(ticker)
    print(f"  Piotroski F: {p}")


def _print_beneish(ticker):
    b = compute_beneish_m(ticker)
    print(f"  Beneish M: {b}")


def main():
    for ticker in CANDIDATES:
        print(f"\n{'=' * 70}\n{ticker}\n{'=' * 70}")
        _print_dupont(ticker)
        _print_altman(ticker)
        _print_piotroski(ticker)
        _print_beneish(ticker)

    print(f"\n{'=' * 70}\nExclusion spot-checks\n{'=' * 70}")
    for ticker, expected_reason in EXCLUSION_SPOT_CHECKS:
        print(f"\n{ticker} (expect '{expected_reason}'):")
        d = compute_dupont(ticker)
        print(f"  DuPont (should still compute): status={d['status']}, business_model_flag={d.get('business_model_flag')}")
        a = compute_altman_z(ticker)
        ok = "OK" if a["status"] == "excluded" and a["business_model_flag"] == expected_reason else "MISMATCH"
        print(f"  Altman: status={a['status']}, business_model_flag={a.get('business_model_flag')} [{ok}]")
        p = compute_piotroski_f(ticker)
        print(f"  Piotroski: status={p['status']}, business_model_flag={p.get('business_model_flag')}")
        b = compute_beneish_m(ticker)
        print(f"  Beneish: status={b['status']}, business_model_flag={b.get('business_model_flag')}")


if __name__ == "__main__":
    main()
