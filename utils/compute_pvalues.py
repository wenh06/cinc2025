"""
Compute Mann-Whitney U p-values for ablation study and model comparison.

Reads from the result CSV files in results/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_ablation_data(csv_path: str | Path) -> pd.DataFrame:
    """Load ablation CSV and aggregate per experiment_name."""
    df = pd.read_csv(csv_path)
    return df


def pairwise_mannwhitney(
    df: pd.DataFrame,
    group_col: str = "experiment_name",
    value_col: str = "challenge_score",
    reference: str | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute pairwise Mann-Whitney U tests between groups.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with one row per repeat.
    group_col : str
        Column identifying the experimental condition.
    value_col : str
        Column with the metric values.
    reference : str, optional
        If given, compare all other groups against this reference group.
        If None, compute all pairwise comparisons.
    alpha : float
        Significance threshold.

    Returns
    -------
    results : dict
        Mapping of (group_a, group_b) -> {"U": float, "p": float, "significant": bool}.
    """
    groups = df[group_col].unique()
    results = {}

    pairs = []
    if reference is not None:
        assert reference in groups, f"Reference group '{reference}' not found in {list(groups)}"
        for g in groups:
            if g != reference:
                pairs.append((reference, g))
    else:
        for i, ga in enumerate(groups):
            for gb in groups[i + 1 :]:
                pairs.append((ga, gb))

    for ga, gb in pairs:
        a = df.loc[df[group_col] == ga, value_col].values
        b = df.loc[df[group_col] == gb, value_col].values
        if len(a) < 3 or len(b) < 3:
            print(f"  Skipping {ga} vs {gb}: insufficient samples (n_a={len(a)}, n_b={len(b)})")
            continue
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        results[(ga, gb)] = {
            "U": u_stat,
            "p": p_val,
            "significant": p_val < alpha,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "n_a": len(a),
            "n_b": len(b),
        }

    return results


def format_p(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ablation_path = repo_root / "results" / "ablation.csv"
    arch_path = repo_root / "results" / "ext-expr-eval-results.csv"

    print("=" * 60)
    print("Statistical Significance Tests (Mann-Whitney U, two-sided)")
    print("=" * 60)

    # --- Ablation Study ---
    if ablation_path.exists():
        print(f"\nAblation Study ({ablation_path.name}):")
        print("-" * 40)
        df_abl = load_ablation_data(ablation_path)
        ref_name = "C_Only_Reliability"
        if ref_name in df_abl["experiment_name"].values:
            results = pairwise_mannwhitney(df_abl, reference=ref_name)
            for (ga, gb), r in results.items():
                sig = "**" if r["significant"] else "n.s."
                mean_diff = r["mean_a"] - r["mean_b"]
                print(f"  {ga} vs {gb}: " f"U={r['U']:.1f}, p={format_p(r['p'])}, " f"diff={mean_diff:+.4f}  {sig}")
        else:
            print(f"  Reference group '{ref_name}' not found. Doing all pairwise.")
            results = pairwise_mannwhitney(df_abl)
            for (ga, gb), r in results.items():
                sig = "**" if r["significant"] else ("*" if r["p"] < 0.05 else "n.s.")
                print(f"  {ga} vs {gb}: " f"U={r['U']:.1f}, p={format_p(r['p'])}  {sig}")
    else:
        print(f"\n  WARNING: {ablation_path} not found, skipping.")

    # --- Architecture Comparison (100% data) ---
    if arch_path.exists():
        print(f"\nArchitecture Comparison ({arch_path.name}, 100% data):")
        print("-" * 40)
        df_arch = load_ablation_data(arch_path)
        df_100 = df_arch[df_arch["subsample_ratio"] == "100%"]
        if len(df_100) > 0:
            results = pairwise_mannwhitney(df_100, group_col="model_arch", value_col="challenge_score")
            for (ga, gb), r in results.items():
                sig = "n.s." if not r["significant"] else "**"
                print(
                    f"  {ga} vs {gb}: "
                    f"U={r['U']:.1f}, p={format_p(r['p'])}, "
                    f"mean {ga}={r['mean_a']:.4f}, mean {gb}={r['mean_b']:.4f}  {sig}"
                )
    else:
        print(f"\n  WARNING: {arch_path} not found, skipping.")

    print()


if __name__ == "__main__":
    main()
