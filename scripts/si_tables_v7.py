"""Generate Supplementary Material tables (S2-S4) in LaTeX form.

Reads benchmark/results/{preset}/{method}/seed{N}.json and writes
LaTeX tabular fragments for the three secondary-metric tables that v7 §6
defers to Supplementary: Δcontiguity, Δbaimu_count, Δbaimu_area_ha.
Matches the format/conventions of tab:bench-slope in the main draft.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections import defaultdict

PRESETS = [
    ("bishan_clone", "2{,}600"),
    ("neijiang_clone", "3{,}700"),
    ("plain_small_cons", "800"),
    ("plain_large_cons", "4{,}500"),
    ("plain_medium_frag", "2{,}600"),
    ("mixed_medium_frag", "2{,}600"),
    ("hilly_small_cons", "800"),
]
METHODS = ["Random", "Greedy", "GA", "PPO", "MPC"]
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
OUT = ROOT.parent.parent / "docs" / "si_tables_v7.tex"
OUT2 = Path(r"D:/test/si_tables_v7.tex")


def load_run(preset: str, method: str, seed: int) -> dict | None:
    p = RESULTS / preset / method / f"seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def fmt(mu: float, sd: float, prec: int) -> str:
    if math.isnan(mu):
        return "---"
    return f"${mu:.{prec}f}{{\\pm}}{sd:.{prec}f}$"


def best_idx(rows: list[float], minimise: bool) -> int:
    """Return the index of the best mean (most negative for slope, otherwise most positive)."""
    arr = [(i, x) for i, x in enumerate(rows) if not math.isnan(x)]
    if not arr:
        return -1
    if minimise:
        return min(arr, key=lambda kv: kv[1])[0]
    return max(arr, key=lambda kv: kv[1])[0]


def render_table(metric_key: str, prec: int, caption: str, label: str,
                 minimise: bool = False) -> str:
    rows = []
    for preset, n_blocks in PRESETS:
        cells_mu = []
        cells_sd = []
        for method in METHODS:
            vals = []
            for seed in range(5):
                r = load_run(preset, method, seed)
                if r is None:
                    continue
                v = r.get(metric_key)
                if v is not None:
                    vals.append(float(v))
            mu, sd = mean_std(vals)
            cells_mu.append(mu)
            cells_sd.append(sd)
        bi = best_idx(cells_mu, minimise=minimise)
        cell_strs = []
        for i, (mu, sd) in enumerate(zip(cells_mu, cells_sd)):
            s = fmt(mu, sd, prec=prec)
            if i == bi:
                s = r"\mathbf{" + s.strip("$") + r"}"
                s = "$" + s + "$"
            cell_strs.append(s)
        rows.append(
            f"\\texttt{{{preset.replace('_', '\\_')}}} ({n_blocks}) & "
            + " & ".join(cell_strs) + r" \\"
        )

    header = (
        r"\begin{table}[htbp]" "\n" r"\centering" "\n" r"\small" "\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        r"\begin{tabular}{lccccc}" "\n" r"\toprule" "\n"
        r"Preset ($n_{\text{blocks}}$) & Random & Greedy & GA & PPO & MPC \\" "\n"
        r"\midrule" "\n"
    )
    footer = "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}"
    return header + "\n".join(rows) + footer


def main() -> None:
    parts = []
    parts.append(render_table(
        metric_key="cont_delta",
        prec=4,
        caption=("Synthetic-benchmark contiguity change ($\\Delta$Contiguity, "
                 "absolute units). Mean $\\pm$ standard deviation across $n{=}5$ seeds. "
                 "Higher (more positive) is better; per-row best in bold."),
        label="tab:si-bench-cont",
        minimise=False,
    ))
    parts.append("")
    parts.append(render_table(
        metric_key="baimu_count_delta",
        prec=1,
        caption=("Synthetic-benchmark large-patch count change "
                 "($\\Delta$\\textit{baimu fang} count, $\\geq 6.67$\\,ha "
                 "qualifying patches gained per episode). Mean $\\pm$ "
                 "standard deviation across $n{=}5$ seeds. Higher is better; "
                 "per-row best in bold."),
        label="tab:si-bench-baimuct",
        minimise=False,
    ))
    parts.append("")
    parts.append(render_table(
        metric_key="baimu_area_delta_ha",
        prec=1,
        caption=("Synthetic-benchmark large-patch area change "
                 "($\\Delta$\\textit{baimu fang} area, ha). Mean $\\pm$ "
                 "standard deviation across $n{=}5$ seeds. The sign of this "
                 "metric is morphology-dependent (see main-text Section~5 "
                 "discussion of the Bishan vs.\\ Neijiang reversal); per-row "
                 "best (most positive) in bold for reference."),
        label="tab:si-bench-baimuha",
        minimise=False,
    ))

    body = "\n".join(parts) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    OUT2.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT}\nWrote {OUT2}\n  ({len(body)} chars)")


if __name__ == "__main__":
    main()
