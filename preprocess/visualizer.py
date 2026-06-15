"""
visualizer.py
-------------
All visualisation functions for the preprocessing pipeline.

Plots produced
--------------
1. ``plot_raw_comparison``   — raw vs filtered vs ICA-cleaned overlay
2. ``plot_psd_comparison``   — PSD curves at each preprocessing stage
3. ``plot_ica_components``   — ICA component topomaps + time-series
4. ``plot_epoch_image``      — epoch image (time × epochs heatmap)
5. ``plot_channel_stds``     — per-channel amplitude std bar chart

All plots are saved as PNG to *figure_dir* and optionally displayed.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(
    fig: plt.Figure,
    save_path: Optional[Path],
    show: bool,
    logger: logging.Logger,
) -> None:
    """Save figure to disk and/or display it, then close."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved: {save_path.name}")
    if show:
        plt.show()
    plt.close(fig)


def _pick_channels(raw: mne.io.BaseRaw, desired: List[str]) -> List[str]:
    """Return the subset of *desired* channels present in *raw*."""
    present = set(raw.ch_names)
    return [ch for ch in desired if ch in present]


# ── 1. Raw comparison plot ────────────────────────────────────────────────────

def plot_raw_comparison(
    raw_orig: mne.io.BaseRaw,
    raw_filtered: mne.io.BaseRaw,
    raw_clean: mne.io.BaseRaw,
    channels: List[str],
    duration: float,
    figure_dir: Path,
    subject_id: str,
    run_id: str,
    show: bool,
    logger: logging.Logger,
) -> None:
    """
    Plot a three-row comparison: original | filtered | ICA-cleaned.

    Parameters
    ----------
    raw_orig     : unfiltered Raw
    raw_filtered : bandpass+notch filtered Raw
    raw_clean    : ICA-cleaned Raw
    channels     : channel names to overlay (those present are used)
    duration     : seconds of signal to display
    figure_dir   : output directory
    subject_id   : used in filename
    run_id       : used in filename
    show         : display interactively
    logger       : logger
    """
    chs = _pick_channels(raw_orig, channels)
    if not chs:
        logger.warning("plot_raw_comparison: none of the desired channels found.")
        return

    sfreq = raw_orig.info["sfreq"]
    n_pts = min(int(duration * sfreq), raw_orig.n_times)
    t = np.arange(n_pts) / sfreq

    raws = [raw_orig, raw_filtered, raw_clean]
    titles = ["Raw (unfiltered)", "Bandpass + Notch filtered", "ICA cleaned"]
    n_rows = len(raws)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 3 * n_rows),
        sharex=True,
    )
    fig.suptitle(
        f"{subject_id} {run_id} — Signal comparison ({duration:.0f}s, "
        f"{len(chs)} channels)",
        fontsize=12, fontweight="bold",
    )

    for ax, raw, title in zip(axes, raws, titles):
        data = raw.get_data(picks=chs)[:, :n_pts] * 1e6  # → µV
        offset = np.arange(len(chs)) * float(np.percentile(np.abs(data), 95)) * 2.5
        for i, (ch, trace) in enumerate(zip(chs, data)):
            ax.plot(t, trace + offset[i], lw=0.6, alpha=0.85)
        ax.set_yticks(offset)
        ax.set_yticklabels(chs, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("µV (offset)")
        ax.grid(axis="x", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    save_path = figure_dir / f"{subject_id}_{run_id}_raw_comparison.png"
    _save_or_show(fig, save_path, show, logger)


# ── 2. PSD comparison ─────────────────────────────────────────────────────────

def plot_psd_comparison(
    raw_orig: mne.io.BaseRaw,
    raw_filtered: mne.io.BaseRaw,
    raw_clean: mne.io.BaseRaw,
    fmin: float,
    fmax: float,
    figure_dir: Path,
    subject_id: str,
    run_id: str,
    show: bool,
    logger: logging.Logger,
) -> None:
    """
    Overlay mean PSD across channels for three preprocessing stages.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"{subject_id} {run_id} — PSD comparison",
        fontsize=12, fontweight="bold",
    )

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    labels = ["Raw", "Filtered", "ICA cleaned"]
    raws   = [raw_orig, raw_filtered, raw_clean]

    for raw, color, label in zip(raws, colors, labels):
        try:
            spec = raw.compute_psd(
                method="welch", fmin=fmin, fmax=fmax, n_fft=2048, verbose=False,
            )
            psds, freqs = spec.get_data(return_freqs=True)
            mean_psd_db = 10 * np.log10(psds.mean(axis=0) + 1e-30)
            ax.plot(freqs, mean_psd_db, color=color, lw=1.5, label=label)
        except Exception as exc:
            logger.warning(f"PSD failed for {label}: {exc}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB re 1 µV²/Hz)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(fmin, fmax)
    plt.tight_layout()

    save_path = figure_dir / f"{subject_id}_{run_id}_psd_comparison.png"
    _save_or_show(fig, save_path, show, logger)


# ── 3. ICA components ─────────────────────────────────────────────────────────

def plot_ica_components(
    ica: ICA,
    raw_hp: mne.io.BaseRaw,
    excluded: List[int],
    figure_dir: Path,
    subject_id: str,
    run_id: str,
    show: bool,
    logger: logging.Logger,
    n_components_plot: int = 20,
) -> None:
    """
    Save ICA topomaps and a properties plot for excluded components.

    Parameters
    ----------
    ica                : fitted ICA
    raw_hp             : high-passed Raw used for fitting
    excluded           : component indices to highlight
    figure_dir         : output directory
    subject_id         : filename prefix
    run_id             : filename prefix
    show               : display interactively
    logger             : logger
    n_components_plot  : how many topomap components to show
    """
    # 3a. Topomaps of all components
    n_plot = min(n_components_plot, ica.n_components_)
    try:
        figs = ica.plot_components(
            picks=list(range(n_plot)),
            show=False,
            title=f"{subject_id} {run_id} ICA topomaps",
        )
        # plot_components can return a list of figures
        figs = figs if isinstance(figs, list) else [figs]
        for i, fig in enumerate(figs):
            save_path = figure_dir / f"{subject_id}_{run_id}_ica_topomaps_{i:02d}.png"
            _save_or_show(fig, save_path, show, logger)
    except Exception as exc:
        logger.warning(f"ICA topomap plot failed: {exc}")

    # 3b. Properties of excluded (artefact) components
    if excluded:
        try:
            figs = ica.plot_properties(
                raw_hp,
                picks=excluded,
                show=False,
            )
            figs = figs if isinstance(figs, list) else [figs]
            for idx, fig in zip(excluded, figs):
                save_path = figure_dir / f"{subject_id}_{run_id}_ica_comp{idx:02d}_props.png"
                _save_or_show(fig, save_path, show, logger)
        except Exception as exc:
            logger.warning(f"ICA properties plot failed: {exc}")

    # 3c. Sources time-series (first n_plot)
    try:
        fig = ica.plot_sources(
            raw_hp,
            picks=list(range(min(n_plot, ica.n_components_))),
            show=False,
            title=f"{subject_id} {run_id} ICA sources",
        )
        save_path = figure_dir / f"{subject_id}_{run_id}_ica_sources.png"
        _save_or_show(fig, save_path, show, logger)
    except Exception as exc:
        logger.warning(f"ICA sources plot failed: {exc}")


# ── 4. Epoch image ────────────────────────────────────────────────────────────

def plot_epoch_image(
    epochs: mne.Epochs,
    channel: str,
    figure_dir: Path,
    subject_id: str,
    run_id: str,
    show: bool,
    logger: logging.Logger,
) -> None:
    """
    Plot an epoch image (epochs × time amplitude heatmap) for one channel.
    """
    if channel not in epochs.ch_names:
        alt = epochs.ch_names[len(epochs.ch_names) // 2]
        logger.warning(f"Channel {channel} not in epochs; using {alt}")
        channel = alt

    try:
        figs = mne.viz.plot_epochs_image(
            epochs,
            picks=[channel],
            show=False,
            title=f"{subject_id} {run_id} — Epoch image ({channel})",
        )
        figs = figs if isinstance(figs, list) else [figs]
        for fig in figs:
            save_path = figure_dir / f"{subject_id}_{run_id}_epoch_image_{channel}.png"
            _save_or_show(fig, save_path, show, logger)
    except Exception as exc:
        logger.warning(f"Epoch image plot failed: {exc}")


# ── 5. Channel standard deviations ───────────────────────────────────────────

def plot_channel_stds(
    raw: mne.io.BaseRaw,
    bad_channels: List[str],
    figure_dir: Path,
    subject_id: str,
    run_id: str,
    show: bool,
    logger: logging.Logger,
) -> None:
    """
    Bar chart of per-channel amplitude std, with bad channels highlighted.
    """
    data = raw.get_data() * 1e6   # → µV
    stds = data.std(axis=1)
    ch_names = raw.ch_names
    colors = ["#d62728" if ch in bad_channels else "#1f77b4" for ch in ch_names]

    fig, ax = plt.subplots(figsize=(max(10, len(ch_names) * 0.18), 4))
    ax.bar(range(len(ch_names)), stds, color=colors, width=0.8)
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_ylabel("Std (µV)")
    ax.set_title(f"{subject_id} {run_id} — Channel std (red = bad)")
    ax.axhline(
        np.median(stds),
        color="green", lw=1.2, ls="--", label=f"Median {np.median(stds):.1f} µV",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()

    save_path = figure_dir / f"{subject_id}_{run_id}_channel_stds.png"
    _save_or_show(fig, save_path, show, logger)


# ── 6. Preprocessing summary dashboard ───────────────────────────────────────

def plot_preprocessing_summary(
    raw_orig: mne.io.BaseRaw,
    raw_clean: mne.io.BaseRaw,
    epochs: mne.Epochs,
    bad_channels: List[str],
    excluded_ica: List[int],
    figure_dir: Path,
    subject_id: str,
    run_id: str,
    show: bool,
    logger: logging.Logger,
) -> None:
    """
    One-page summary dashboard combining key stats in a single figure.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Preprocessing Summary — {subject_id} {run_id}",
        fontsize=14, fontweight="bold",
    )

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # -- Panel A: Channel stds (before cleaning) --
    ax_a = fig.add_subplot(gs[0, :])
    data_raw = raw_orig.get_data() * 1e6
    stds = data_raw.std(axis=1)
    colors = ["#d62728" if ch in bad_channels else "#aec7e8"
              for ch in raw_orig.ch_names]
    ax_a.bar(range(len(raw_orig.ch_names)), stds, color=colors, width=0.9)
    ax_a.set_xticks(range(len(raw_orig.ch_names)))
    ax_a.set_xticklabels(raw_orig.ch_names, rotation=90, fontsize=5)
    ax_a.set_ylabel("Std (µV)")
    ax_a.set_title("Per-channel Std (red = bad)")
    ax_a.axhline(np.median(stds), color="g", ls="--", lw=1.0,
                 label=f"Median {np.median(stds):.1f} µV")
    ax_a.legend(fontsize=7)

    # -- Panel B: PSD overlay --
    ax_b = fig.add_subplot(gs[1, :2])
    for raw, color, label in [
        (raw_orig, "#d62728", "Raw"),
        (raw_clean, "#2ca02c", "Cleaned"),
    ]:
        try:
            spec = raw.compute_psd(method="welch", fmin=0.5, fmax=50.0,
                                   n_fft=2048, verbose=False)
            psds, freqs = spec.get_data(return_freqs=True)
            ax_b.plot(freqs, 10 * np.log10(psds.mean(axis=0) + 1e-30),
                      color=color, lw=1.5, label=label)
        except Exception:
            pass
    ax_b.set_xlabel("Frequency (Hz)")
    ax_b.set_ylabel("Power (dB)")
    ax_b.set_title("Mean PSD: Raw vs Cleaned")
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    # -- Panel C: Stats text box --
    ax_c = fig.add_subplot(gs[1, 2])
    ax_c.axis("off")
    stats_text = (
        f"Subject: {subject_id}\n"
        f"Run:     {run_id}\n\n"
        f"Channels:      {len(raw_orig.ch_names)}\n"
        f"Bad channels:  {len(bad_channels)}\n"
        f"ICA excluded:  {len(excluded_ica)}\n\n"
        f"Sfreq:   {raw_orig.info['sfreq']:.0f} Hz\n"
        f"Duration: {raw_orig.n_times / raw_orig.info['sfreq']:.1f}s\n\n"
        f"Epochs kept: {len(epochs) if epochs is not None else 'N/A'}\n"
    )
    ax_c.text(
        0.05, 0.95, stats_text,
        transform=ax_c.transAxes,
        va="top", fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
    )
    ax_c.set_title("Run Info")

    # -- Panel D: Epoch amplitude distribution --
    ax_d = fig.add_subplot(gs[2, :])
    if epochs is not None and len(epochs) > 0:
        ep_data = epochs.get_data() * 1e6  # (n_ep, n_ch, n_t)
        pk2pk = ep_data.max(axis=2) - ep_data.min(axis=2)  # (n_ep, n_ch)
        mean_pk2pk = pk2pk.mean(axis=1)                      # per epoch
        ax_d.hist(mean_pk2pk, bins=40, color="#1f77b4", edgecolor="white", lw=0.3)
        ax_d.axvline(mean_pk2pk.mean(), color="orange", lw=1.5,
                     label=f"Mean {mean_pk2pk.mean():.1f} µV")
        ax_d.set_xlabel("Mean peak-to-peak amplitude (µV)")
        ax_d.set_ylabel("Count")
        ax_d.set_title("Epoch amplitude distribution (mean over channels)")
        ax_d.legend(fontsize=8)
    else:
        ax_d.text(0.5, 0.5, "No valid epochs", ha="center", va="center",
                  transform=ax_d.transAxes, fontsize=12)

    save_path = figure_dir / f"{subject_id}_{run_id}_summary.png"
    _save_or_show(fig, save_path, show, logger)
