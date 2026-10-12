"""Turn a results JSON into the Markdown tables and the learning-curve SVG used in the README."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np

LABEL_MODEL_ROWS = (
    ("majority_vote", "Majority vote"),
    ("dawid_skene", "Dawid-Skene"),
    ("metal", "MeTaL (balance from valid labels)"),
    ("metal_estimated_balance", "MeTaL (estimated balance)"),
    ("flyingsquid", "FlyingSquid (balance from valid labels)"),
    ("flyingsquid_estimated_balance", "FlyingSquid (estimated balance)"),
    ("snorkel", "Snorkel LabelModel (reference)"),
)
STRATEGY_ROWS = (
    ("random", "Random"),
    ("conflict_entropy", "LF-conflict entropy"),
    ("posterior_margin", "Posterior margin"),
    ("coreset", "Core-set (k-center)"),
    ("random_gold_labels_only", "Random, human labels only"),
)


def _pct(stat: dict[str, float], with_std: bool = True) -> str:
    text = f"{100 * stat['mean']:.1f}"
    if with_std and stat.get("std", 0.0) >= 0.0005:
        text += f" ± {100 * stat['std']:.1f}"
    return text


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def label_model_table(res: dict[str, Any], section: str) -> str:
    """``section`` is ``label_model`` or ``end_model``; cells are accuracy / F1 in %."""
    datasets = list(res["label_models"])
    rows = []
    for key, title in LABEL_MODEL_ROWS:
        if not any(key in res["label_models"][d] for d in datasets):
            continue
        cells = [title]
        for d in datasets:
            entry = res["label_models"][d].get(key)
            if entry is None:
                cells.append("not run")
            else:
                cells.append(
                    f"{_pct(entry[f'{section}_accuracy'])} / {_pct(entry[f'{section}_f1'])}"
                )
        rows.append(cells)
    return _table(["Model", *datasets], rows)


def fit_time_table(res: dict[str, Any]) -> str:
    datasets = list(res["label_models"])
    rows = []
    for key, title in LABEL_MODEL_ROWS:
        if not any(key in res["label_models"][d] for d in datasets):
            continue
        rows.append(
            [title]
            + [
                f"{res['label_models'][d][key]['fit_seconds']['mean']:.2f}"
                if key in res["label_models"][d]
                else "not run"
                for d in datasets
            ]
        )
    rows.append(
        ["Class-balance estimation (tensor)"]
        + [f"{res['class_balance'][d]['estimation_seconds']:.2f}" for d in datasets]
    )
    return _table(["Fit time (s)", *datasets], rows)


def balance_table(res: dict[str, Any]) -> str:
    rows = []
    for d, b in res["class_balance"].items():
        rows.append(
            [
                d,
                ", ".join(f"{x:.3f}" for x in b["true_train"]),
                ", ".join(f"{x:.3f}" for x in b["estimated"]),
                f"{b['estimated_l1_error']:.3f}",
                f"{b['valid_l1_error']:.3f}",
            ]
        )
    return _table(
        [
            "Dataset",
            "True train balance",
            "Tensor estimate",
            "L1 error (estimate)",
            "L1 error (valid labels)",
        ],
        rows,
    )


def aulc_table(res: dict[str, Any]) -> str:
    datasets = list(res["active_learning"])
    rows = []
    for key, title in STRATEGY_ROWS:
        cells = [title]
        for d in datasets:
            entry = res["active_learning"][d].get(key)
            if entry is None:
                cells.append("not run")
                continue
            cell = _pct(entry["nAULC"]["end_accuracy"])
            if key != "random":
                diff = entry["nAULC_accuracy_minus_random"]
                sign = "+" if diff["mean"] >= 0 else "-"
                cell += (
                    f" ({sign}{abs(100 * diff['mean']):.1f}, "
                    f"{diff['seeds_better']}/{diff['seeds']} seeds)"
                )
            cells.append(cell)
        rows.append(cells)
    return _table(["Strategy", *datasets], rows)


def curve_table(res: dict[str, Any], dataset: str) -> str:
    al = res["active_learning"][dataset]
    budgets = al["random"]["curve"]["n_labels"]
    rows = []
    for key, title in STRATEGY_ROWS:
        if key not in al:
            continue
        curve = al[key]["curve"]
        by_budget = dict(zip(curve["n_labels"], range(len(curve["n_labels"])), strict=True))
        cells = [title]
        for b in budgets:
            i = by_budget.get(b)
            cells.append(
                "-"
                if i is None
                else _pct(
                    {
                        "mean": curve["end_accuracy"]["mean"][i],
                        "std": curve["end_accuracy"]["std"][i],
                    }
                )
            )
        rows.append(cells)
    return _table(["End-model accuracy (%) at labels", *map(str, budgets)], rows)


def synthetic_table(res: dict[str, Any]) -> str:
    rows = []
    for setting, s in res["synthetic_recovery"].items():
        mae = s["lf_accuracy_mae"]
        rows.append(
            [
                f"{setting} (n={s['n']}, m={s['m']})",
                *(
                    f"{mae[name]['mean']:.4f}" if name in mae else "not run"
                    for name in ("dawid_skene", "metal", "flyingsquid", "snorkel")
                ),
                f"{s['tensor_balance_l1_error']['mean']:.4f}",
            ]
        )
    return _table(
        [
            "Setting",
            "Dawid-Skene MAE",
            "MeTaL MAE",
            "FlyingSquid MAE",
            "Snorkel MAE",
            "Balance L1 error (tensor)",
        ],
        rows,
    )


def markdown_report(res: dict[str, Any]) -> str:
    m = res["machine"]
    parts = [
        f"Suite `{res['suite']}`, {res['date']}, commit `{res['git_sha'][:12]}`, "
        f"{m['cpu']}, {m['physical_cores']} cores, {m['ram_gb']} GB RAM, {m['os']}, "
        f"Python {m['python']}. Wall time {res['wall_seconds'] / 60:.1f} min.",
        "### Label models on the test split (accuracy / F1, %)",
        label_model_table(res, "label_model"),
        "### End model trained on each label model's soft labels (accuracy / F1, %)",
        label_model_table(res, "end_model"),
        "### Fit time",
        fit_time_table(res),
        "### Class balance",
        balance_table(res),
        "### Active learning: normalised AULC of end-model accuracy (%)",
        aulc_table(res),
    ]
    for d in res["active_learning"]:
        parts += [f"#### {d}", curve_table(res, d)]
    if res.get("synthetic_recovery"):
        parts += ["### Parameter recovery on synthetic label matrices", synthetic_table(res)]
    return "\n\n".join(parts) + "\n"


# Learning-curve figure --------------------------------------------------------------------------

# Categorical slots 1-3 in fixed order; random is the reference, drawn in neutral gray. The
# human-labels-only curve stays in the tables: on the same axis it would flatten the others.
SERIES = (
    ("random", "Random", "#8a8984", "6 4", "circle"),
    ("conflict_entropy", "LF-conflict entropy", "#2a78d6", "", "square"),
    ("posterior_margin", "Posterior margin", "#eb6834", "", "triangle"),
    ("coreset", "Core-set", "#1baf7a", "", "diamond"),
)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PANEL_W, PANEL_H, LEGEND_ROW = 300, 220, 22
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 52, 14, 34, 40


def _el(tag: str, text: str | None = None, **attrs: object) -> str:
    rendered = " ".join(
        f'{k.rstrip("_").replace("_", "-")}="{v:.1f}"'
        if isinstance(v, float)
        else f'{k.rstrip("_").replace("_", "-")}="{escape(str(v))}"'
        for k, v in attrs.items()
    )
    return f"<{tag} {rendered}/>" if text is None else f"<{tag} {rendered}>{escape(text)}</{tag}>"


def _marker(shape: str, x: float, y: float, color: str) -> str:
    r = 4.0
    style = {"fill": color, "stroke": SURFACE, "stroke_width": 1.5}
    if shape == "square":
        return _el("rect", x=x - r, y=y - r, width=2 * r, height=2 * r, **style)
    if shape == "triangle":
        pts = [(x, y - r - 1), (x - r - 0.5, y + r), (x + r + 0.5, y + r)]
    elif shape == "diamond":
        pts = [(x, y - r - 1), (x + r + 1, y), (x, y + r + 1), (x - r - 1, y)]
    else:
        return _el("circle", cx=x, cy=y, r=r, **style)
    return _el("polygon", points=" ".join(f"{a:.1f},{b:.1f}" for a, b in pts), **style)


def _axis_range(lows: list[float], highs: list[float]) -> tuple[float, float]:
    """Snap the y-range outward to multiples of 0.05 (5 accuracy points)."""
    return max(0.0, int(min(lows) * 20) / 20), min(1.0, (int(max(highs) * 20) + 1) / 20)


def _panel(name: str, al: dict[str, Any], ox: float, oy: float, metric: str) -> list[str]:
    series = [(s, al[s[0]]["curve"]) for s in SERIES if s[0] in al]
    xs_all = sorted({x for _, c in series for x in c["n_labels"]})
    bands = [
        (m - sd, m + sd)
        for _, c in series
        for m, sd in zip(c[metric]["mean"], c[metric]["std"], strict=True)
    ]
    y0, y1 = _axis_range([lo for lo, _ in bands], [hi for _, hi in bands])
    x1 = float(xs_all[-1])
    iw, ih = PANEL_W - PAD_LEFT - PAD_RIGHT, PANEL_H - PAD_TOP - PAD_BOTTOM

    def sx(x: float) -> float:
        return ox + PAD_LEFT + x / x1 * iw

    def sy(y: float) -> float:
        return oy + PAD_TOP + (1 - (y - y0) / (y1 - y0)) * ih

    out = [_el("text", name, x=ox + PAD_LEFT, y=oy + 20, font_size=13, font_weight=600, fill=INK)]
    tick = 0.05 if y1 - y0 <= 0.3 else 0.1
    first = np.ceil(y0 / tick - 1e-9) * tick
    for yv in np.arange(first, y1 + 1e-9, tick):
        out.append(_el("line", x1=sx(0), x2=sx(x1), y1=sy(yv), y2=sy(yv), stroke=GRID))
        label = f"{100 * yv:.0f}"
        out.append(
            _el(
                "text",
                label,
                x=sx(0) - 6,
                y=sy(yv) + 4,
                font_size=10,
                text_anchor="end",
                fill=MUTED,
            )
        )
    for xv in xs_all:
        out.append(
            _el(
                "text",
                str(xv),
                x=sx(xv),
                y=sy(y0) + 14,
                font_size=10,
                text_anchor="middle",
                fill=MUTED,
            )
        )
    out.append(
        _el(
            "text",
            "human labels",
            x=ox + PAD_LEFT + iw / 2,
            y=oy + PANEL_H - 8,
            font_size=10,
            text_anchor="middle",
            fill=MUTED,
        )
    )
    for (_key, _title, color, dash, shape), curve in series:
        xs, mean, std = curve["n_labels"], curve[metric]["mean"], curve[metric]["std"]
        upper = [f"{sx(x):.1f},{sy(m + sd):.1f}" for x, m, sd in zip(xs, mean, std, strict=True)]
        lower = [f"{sx(x):.1f},{sy(m - sd):.1f}" for x, m, sd in zip(xs, mean, std, strict=True)]
        out.append(
            _el("polygon", points=" ".join(upper + lower[::-1]), fill=color, fill_opacity=0.12)
        )
        line = " ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m in zip(xs, mean, strict=True))
        out.append(
            _el(
                "polyline",
                points=line,
                fill="none",
                stroke=color,
                stroke_width=2,
                stroke_dasharray=dash or "none",
            )
        )
        out += [_marker(shape, sx(x), sy(m), color) for x, m in zip(xs, mean, strict=True)]
    return out


def learning_curve_svg(res: dict[str, Any], metric: str = "end_accuracy") -> str:
    """Small multiples, one panel per dataset: mean over seeds with a +-1 std band."""
    datasets = list(res["active_learning"])
    cols = min(3, len(datasets))
    rows = -(-len(datasets) // cols)
    width = cols * PANEL_W
    # Legend items flow left to right and wrap, so a single narrow panel keeps them all.
    items: list[tuple[float, float, tuple[str, str, str, str, str]]] = []
    lx, ly = 12.0, 18.0
    for entry in SERIES:
        item_w = 46 + 6.5 * len(entry[1])
        if lx > 12.0 and lx + item_w > width - 8:
            lx, ly = 12.0, ly + LEGEND_ROW
        items.append((lx, ly, entry))
        lx += item_w
    legend_h = ly + LEGEND_ROW
    height = rows * PANEL_H + legend_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        _el("rect", width=width, height=height, fill=SURFACE),
    ]
    for lx, ly, (_key, title, color, dash, shape) in items:
        out.append(
            _el(
                "line",
                x1=lx,
                y1=ly,
                x2=lx + 22,
                y2=ly,
                stroke=color,
                stroke_width=2,
                stroke_dasharray=dash or "none",
            )
        )
        out.append(_marker(shape, lx + 11, ly, color))
        out.append(_el("text", title, x=lx + 28, y=ly + 4, font_size=12, fill=INK))
    for n, name in enumerate(datasets):
        ox, oy = (n % cols) * PANEL_W, legend_h + (n // cols) * PANEL_H
        out += _panel(name, res["active_learning"][name], float(ox), float(oy), metric)
    out.append("</svg>")
    return "\n".join(out) + "\n"
