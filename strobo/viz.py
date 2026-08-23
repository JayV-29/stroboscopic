"""Figures.  One palette, one style, every plot reads as one system.

Categorical slots (fixed order, never cycled):
  ours=blue, fixed-rate=orange, send-on-delta=aqua, kalman=yellow,
  imu-gated=magenta, learned-threshold=green, ablations=violet/red.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec

SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {
    "ours": "#2a78d6", "fixed_rate": "#eb6834", "send_on_delta": "#1baf7a", "kalman": "#eda100",
    "imu_gated": "#e87ba4", "learned_threshold": "#008300", "ours_no_fisher": "#4a3aa7",
    "ours_no_fallback": "#e34948",
}
LABELS = {
    "ours": "Stroboscopic (ours)", "fixed_rate": "Fixed rate", "send_on_delta": "Send-on-delta",
    "kalman": "Event-triggered Kalman", "imu_gated": "IMU-gated", "learned_threshold": "Learned threshold",
    "ours_no_fisher": "Ours – no Fisher", "ours_no_fallback": "Ours – no fallback",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "medium",
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
        "lines.markersize": 6, "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path)
    return fig


# --------------------------------------------------------------------------- #
# 1. accuracy-energy curves, rest vs motion
# --------------------------------------------------------------------------- #
def plot_curves(df, target: str = "hr", methods=None, path=None, title=None, ylim=None,
                x="bursts_per_beat", modes=("rest", "motion"), success=None):
    """df columns: method, mode, bursts_per_beat, mae_<target> (already averaged over folds
    or raw per fold; both fine – we plot the Pareto front of mean-over-folds points)."""
    from .eval import pareto_front
    use_style()
    methods = methods or [m for m in SERIES if m in set(df.method)]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    col = f"mae_{target}"
    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        for m in methods:
            d = sub[sub.method == m]
            if d.empty or d[col].isna().all():
                continue
            g = d.groupby("setting").agg(bursts_per_beat=(x, "mean"), mae=(col, "mean"),
                                         sd=(col, "std")).reset_index()
            f = pareto_front(g, "bursts_per_beat", "mae").sort_values("bursts_per_beat")
            c = SERIES[m]
            ax.plot(g.bursts_per_beat, g.mae, "o", color=c, alpha=0.35, ms=4, zorder=2)
            ax.plot(f.bursts_per_beat, f.mae, "-o", color=c, label=LABELS[m], zorder=3 if m != "ours" else 5,
                    lw=2.6 if m == "ours" else 1.8, ms=6 if m == "ours" else 4)
            if f.sd.notna().any():
                ax.fill_between(f.bursts_per_beat, f.mae - f.sd.fillna(0), f.mae + f.sd.fillna(0),
                                color=c, alpha=0.10, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("bursts per beat  (log)")
        ax.set_title(f"{mode.capitalize()}")
        if success:
            ax.axvline(1.0, color=AXIS, lw=1, ls=":")
            ax.axhline(success, color=AXIS, lw=1, ls=":")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(f"{target.upper()} MAE" + (" (bpm)" if target in ("hr", "rr", "cadence") else " (ms)"))
    axes[-1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(title or f"{target.upper()} error vs. sampling cost", x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. phase histogram (polar + linear)
# --------------------------------------------------------------------------- #
def plot_phase_histogram(hists: dict, path=None, title="Where in the cardiac cycle does the model sample?"):
    """hists: {label: phase_histogram-dict} e.g. {'rest': ..., 'motion': ...}."""
    use_style()
    fig = plt.figure(figsize=(10, 4.2))
    gs = GridSpec(1, 2, width_ratios=[1, 1.5], figure=fig)
    axp = fig.add_subplot(gs[0], projection="polar")
    axl = fig.add_subplot(gs[1])
    colors = [SERIES["ours"], SERIES["fixed_rate"], SERIES["send_on_delta"], SERIES["kalman"]]
    for (lab, h), c in zip(hists.items(), colors):
        centers = 0.5 * (h["edges"][1:] + h["edges"][:-1])
        w = h["edges"][1] - h["edges"][0]
        axp.bar(centers, h["density"], width=w, color=c, alpha=0.55, edgecolor=SURFACE, lw=0.8,
                label=f"{lab} (R={h['R']:.2f}, n={h['n']})")
        axl.step(np.r_[h["edges"][0], centers, h["edges"][-1]] / (2 * np.pi),
                 np.r_[h["density"][0], h["density"], h["density"][-1]], where="mid", color=c, label=lab)
        axl.fill_between(centers / (2 * np.pi), h["density"], step="mid", color=c, alpha=0.15, lw=0)
    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    axp.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    axp.set_xticklabels(["R-peak", "", "¼", "", "½", "", "¾", ""], color=MUTED)
    axp.set_yticklabels([]); axp.grid(color=GRID)
    axp.set_title("polar", color=INK2, pad=12)
    axl.axhline(1 / (2 * np.pi), color=AXIS, ls=":", lw=1)
    axl.text(0.99, 1 / (2 * np.pi), "uniform", ha="right", va="bottom", color=MUTED, fontsize=8)
    axl.set_xlabel("cardiac phase (fraction of RR interval after R-peak)")
    axl.set_ylabel("density of fired bursts")
    axl.set_xlim(0, 1)
    axl.legend(loc="upper right")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 3. example trace
# --------------------------------------------------------------------------- #
def plot_trace(tr: dict, path=None, title="One 8-second window through the stroboscopic sampler", target_idx=0):
    use_style()
    t = tr["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.4, 0.9, 1.0]})
    ax = axes[0]
    for j, c in enumerate([SEQ_BLUE[2], SEQ_BLUE[4], SEQ_BLUE[6]][: tr["cheap"].shape[1]]):
        ax.plot(t, tr["cheap"][:, j], color=c, lw=1.2)
    ax.set_ylabel("wrist ACC\n(always on)")
    ax = axes[1]
    k = tr["expensive"].shape[1]
    tt = np.repeat(t, k) + np.tile(np.arange(k) / (k * tr["fs"]), t.size)
    x = tr["expensive"].reshape(-1)
    ax.plot(tt, x, color=GRID, lw=1.0, label="PPG (hidden)")
    obs = np.repeat(tr["obs"], k) > 0.5
    ax.plot(np.where(obs, tt, np.nan), np.where(obs, x, np.nan), color=SERIES["ours"], lw=2.4, label="observed bursts")
    for b in np.where(tr["beats"] > 0)[0]:
        ax.axvline(t[b], color=SERIES["fixed_rate"], lw=0.9, alpha=0.7)
    ax.plot([], [], color=SERIES["fixed_rate"], lw=0.9, label="ECG R-peak")
    ax.set_ylabel("PPG"); ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax = axes[2]
    if "p_policy" in tr:
        ax.plot(t, tr["p_policy"], color=SERIES["ours"], lw=1.4, label="P(fire)")
    ax.plot(t, tr["coh"][:, 0], color=SERIES["learned_threshold"], lw=1.4, label="prediction coherence")
    ax.plot(t, tr["gate"], color=SERIES["ours_no_fallback"], lw=1.2, ls="--", label="fallback gate")
    ax.set_ylim(-0.02, 1.05); ax.set_ylabel("policy"); ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax = axes[3]
    name = tr["target_names"][target_idx]
    ax.plot(t, tr["targets"][:, target_idx], color=INK2, lw=1.6, label=f"{name} (ECG truth)")
    ax.plot(t, tr["pred"][:, target_idx], color=SERIES["ours"], lw=2.0, label=f"{name} (decoded)")
    ax.fill_between(t, tr["pred"][:, target_idx] - 1.96 * tr["std"][:, target_idx],
                    tr["pred"][:, target_idx] + 1.96 * tr["std"][:, target_idx], color=SERIES["ours"], alpha=0.12, lw=0)
    ax.set_ylabel(name.upper()); ax.set_xlabel("time (s)"); ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 4. ablation / cross-task table as an image
# --------------------------------------------------------------------------- #
def table_figure(df, path=None, title="", col_width=1.6, fontsize=9.5, fmt="{:.2f}"):
    use_style()
    nrows, ncols = df.shape
    fig, ax = plt.subplots(figsize=(1.3 + col_width * ncols, 0.45 * (nrows + 2)))
    ax.axis("off")
    cells = [[(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)) for v in row]
             for row in df.values]
    tab = ax.table(cellText=cells, colLabels=list(df.columns), rowLabels=list(df.index), loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(fontsize); tab.scale(1, 1.4)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(SEQ_BLUE[0]); cell.set_text_props(color=INK, weight="medium")
        elif c == -1:
            cell.set_facecolor(PAGE); cell.set_text_props(color=INK2)
        else:
            cell.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, pad=8)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 5. architecture diagram (vector-style, drawn with patches)
# --------------------------------------------------------------------------- #
def architecture_figure(path=None, n_params=None, macs=None):
    use_style()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.4)

    def box(x, y, w, h, title, sub, color, text=INK):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                            fc=color, ec="none"))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=10.5, weight="medium", color=text)
        ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center", fontsize=8.2, color=text, linespacing=1.4)

    def arrow(x0, y0, x1, y1, label=None, color=INK2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.16, label, ha="center", fontsize=8, color=color)

    box(0.2, 1.4, 1.9, 1.7, "Wrist IMU", "32 Hz · always on\n3-axis ACC", "#eef0f2")
    box(2.6, 1.4, 2.1, 1.7, "Conv encoder", "3 x causal conv, 16 ch\n2.7 s receptive field\nout: 32-d feature", SEQ_BLUE[0])
    box(5.2, 1.4, 2.5, 1.7, "Oscillator bank", "12 Kuramoto oscillators\nθ, ω, a  +  templates w(θ)\ncoherence r, Fisher(θ)", SEQ_BLUE[1])
    box(8.2, 2.5, 1.9, 1.5, "Sampling head", "(cos θ, sin θ, r, last burst)\nP(fire), Gumbel straight-through", "#fde6da")
    box(8.2, 0.3, 1.9, 1.5, "Decoder", "2 × 64 MLP\nHR · HRV · RR · cadence\nGaussian NLL", "#dcefe6")
    box(10.6, 1.4, 1.3, 1.7, "Expensive\nsensor", "PPG / resp / ECG\n125 ms bursts", "#eef0f2")
    arrow(2.1, 2.25, 2.6, 2.25)
    arrow(4.7, 2.25, 5.2, 2.25)
    arrow(7.7, 2.6, 8.2, 3.1, "phase")
    arrow(7.7, 1.9, 8.2, 1.2, "state")
    arrow(10.1, 3.25, 10.6, 2.9, "fire", SERIES["fixed_rate"])
    arrow(10.6, 1.8, 10.1, 1.0, "burst", SERIES["ours"])
    ax.annotate("", xy=(6.5, 1.4), xytext=(11.0, 1.4), arrowprops=dict(arrowstyle="-|>", color=SERIES["ours"], lw=1.4,
                                                                          connectionstyle="arc3,rad=-0.22"))
    ax.text(8.0, 0.55, "burst feeds back into the oscillators (phase re-sync)", fontsize=8, color=SERIES["ours"], ha="center")
    ax.text(0.2, 3.95, "Coherence r below a learned threshold opens a fallback gate to dense sampling (part of the model, not a rule)",
            fontsize=8.5, color=SERIES["ours_no_fallback"])
    foot = "Objective:  NLL(targets) + λe·mean(fire) − λf·mean(fire·Fisher) + λc·gate + λr·template recon"
    if n_params:
        foot += f"     |     {n_params/1000:.1f} k params · {macs} MACs / tick"
    ax.text(0.2, 0.12, foot, fontsize=8.5, color=INK2)
    ax.set_title("Stroboscopic sensing: one tick of the pipeline", loc="left", fontsize=12)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 6. animation: oscillators + sampling over a window
# --------------------------------------------------------------------------- #
def make_animation(tr: dict, path: str, fps: int = 16, stride: int = 2, max_seconds: float = 8.0):
    """GIF: left = oscillator phases on a circle (size ∝ amplitude, task group brighter),
    right = PPG being revealed burst by burst with decoded HR.  Returns path."""
    import imageio.v2 as imageio
    from PIL import Image
    use_style()
    T = min(tr["t"].size, int(max_seconds * tr["fs"]))
    k = tr["expensive"].shape[1]
    x = tr["expensive"].reshape(-1)
    tt = np.repeat(tr["t"], k) + np.tile(np.arange(k) / (k * tr["fs"]), tr["t"].size)
    obs = np.repeat(tr["obs"], k) > 0.5
    theta = tr.get("theta")
    amp = tr.get("amp")
    frames = []
    for ti in range(0, T, stride):
        fig = plt.figure(figsize=(10, 3.6), dpi=90)
        gs = GridSpec(1, 2, width_ratios=[1, 2.2], figure=fig)
        axc = fig.add_subplot(gs[0]); axs = fig.add_subplot(gs[1])
        axc.set_aspect("equal"); axc.set_xlim(-1.3, 1.3); axc.set_ylim(-1.3, 1.3); axc.axis("off")
        axc.add_patch(patches.Circle((0, 0), 1.0, fc="none", ec=AXIS, lw=1.2))
        if theta is not None:
            th, a = theta[ti], amp[ti] if amp is not None else np.ones(theta.shape[1])
            axc.scatter(np.cos(th), np.sin(th), s=40 + 160 * a, c=[SEQ_BLUE[min(6, 1 + int(5 * v))] for v in a],
                        edgecolors=SURFACE, lw=1.0, zorder=3)
            z = np.exp(1j * th).mean()
            axc.annotate("", xy=(z.real, z.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=SERIES["ours_no_fallback"], lw=2))
        rp = tr["ref_phase"][ti]
        if np.isfinite(rp):
            axc.plot([0, 1.15 * np.cos(rp)], [0, 1.15 * np.sin(rp)], color=SERIES["fixed_rate"], lw=1.2, ls=":")
        axc.text(0, -1.25, f"r = {tr['coh'][ti, 0]:.2f}   gate = {tr['gate'][ti]:.2f}", ha="center", fontsize=9, color=INK2)
        axc.set_title("latent oscillators", fontsize=10, color=INK2)
        n = ti * k
        axs.plot(tt[:n], x[:n], color=GRID, lw=1.0)
        axs.plot(np.where(obs[:n], tt[:n], np.nan), np.where(obs[:n], x[:n], np.nan), color=SERIES["ours"], lw=2.4)
        for b in np.where(tr["beats"][:ti] > 0)[0]:
            axs.axvline(tr["t"][b], color=SERIES["fixed_rate"], lw=0.8, alpha=0.6)
        if tr["fire"][ti] > 0.5:
            axs.axvspan(tr["t"][ti], tr["t"][ti] + 0.125, color=SERIES["ours"], alpha=0.25, lw=0)
        axs.set_xlim(0, tr["t"][T - 1]); axs.set_ylim(x[:T * k].min() - 0.3, x[:T * k].max() + 0.6)
        axs.set_xlabel("time (s)"); axs.set_yticks([])
        hr_p, hr_t = tr["pred"][ti, 0], tr["targets"][ti, 0]
        nb = tr["fire"][:ti].sum() / max(tr["beats"][:ti].sum(), 1)
        axs.set_title(f"PPG revealed by bursts   ·   decoded {tr['target_names'][0].upper()} {hr_p:5.1f}  (truth {hr_t:5.1f})   ·   "
                      f"{nb:.2f} bursts/beat", fontsize=10, color=INK2, loc="left")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img); plt.close(fig)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    return path


# --------------------------------------------------------------------------- #
# 7. poster: compose the saved PNGs into a single summary image
# --------------------------------------------------------------------------- #
def compose_poster(paths: list[str], out: str, cols: int = 2, width: int = 2200, title: str = "Stroboscopic sensing — results"):
    from PIL import Image, ImageDraw, ImageFont
    imgs = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
    if not imgs:
        return None
    cw = width // cols
    scaled = []
    for im in imgs:
        s = cw / im.width
        scaled.append(im.resize((cw, int(im.height * s)), Image.LANCZOS))
    rows = [scaled[i:i + cols] for i in range(0, len(scaled), cols)]
    hdr = 120
    H = hdr + sum(max(im.height for im in r) for r in rows) + 40 * len(rows)
    canvas = Image.new("RGB", (width, H), PAGE)
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 54)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 54)
        except Exception:
            font = ImageFont.load_default()
    d.text((40, 30), title, fill=INK, font=font)
    y = hdr
    for r in rows:
        for j, im in enumerate(r):
            canvas.paste(im, (j * cw, y))
        y += max(im.height for im in r) + 40
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    canvas.save(out, quality=92)
    return out
