"""
summary_visualiser.py
---------------------
Generates QC figures for cleaned epochs.
Processes S001-S005 (R01) automatically.
"""

import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # <-- forces non‑interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path.home() / "24BCE1822" / "summary_visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

subjects = ["S001", "S002", "S003", "S004", "S005"]

summary_rows = []

for subject in subjects:

    print("\n" + "=" * 60)
    print(f"Processing {subject}")
    print("=" * 60)

    epochs_path = Path(
        f"~/24BCE1822/preprocessed/{subject}/{subject}_R01_baseline_eyes_open_epo.fif"
    ).expanduser()

    if not epochs_path.exists():
        print(f"Missing: {epochs_path}")
        continue

    subject_dir = OUTPUT_DIR / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    epochs = mne.read_epochs(epochs_path, preload=True)

    print(
        f"Loaded {len(epochs)} epochs, "
        f"{len(epochs.ch_names)} channels."
    )

    # ============================================================
    # 1. Butterfly Plot
    # ============================================================

    fig1 = epochs.plot(
        n_channels=len(epochs.ch_names),
        scalings="auto",
        title=f"{subject} - Butterfly Plot",
        show=False
    )

    fig1.savefig(
        subject_dir / "butterfly_cleaned.png",
        dpi=150
    )

    plt.close(fig1)

    # ============================================================
    # 2. Evoked Average
    # ============================================================

    evoked = epochs.average()

    fig2 = evoked.plot(
        spatial_colors=True,
        gfp=True,
        show=False
    )

    fig2.savefig(
        subject_dir / "evoked_average.png",
        dpi=150
    )

    plt.close(fig2)

    # ============================================================
    # 3. Topomap at Peak GFP
    # ============================================================

    gfp = evoked.data.std(axis=0)

    peak_idx = np.argmax(gfp)

    peak_time = evoked.times[peak_idx]

    fig3 = evoked.plot_topomap(
        times=peak_time,
        size=2,
        show=False
    )

    fig3.suptitle(
        f"{subject} - Topomap at {peak_time:.3f}s"
    )

    fig3.savefig(
        subject_dir / "topomap_peak.png",
        dpi=150
    )

    plt.close(fig3)

    # ============================================================
    # 4. Epoch Image
    # ============================================================

    pick = "C3" if "C3" in epochs.ch_names else epochs.ch_names[0]

    figs = mne.viz.plot_epochs_image(
        epochs,
        picks=[pick],
        show=False
    )

    if isinstance(figs, list):
        for fig in figs:
            fig.savefig(
                subject_dir / f"epochs_image_{pick}.png",
                dpi=150
            )
            plt.close(fig)
    else:
        figs.savefig(
            subject_dir / f"epochs_image_{pick}.png",
            dpi=150
        )
        plt.close(figs)

    # ============================================================
    # 5. PSD
    # ============================================================

    fig5 = epochs.plot_psd(
        average=True,
        fmin=0.5,
        fmax=50,
        dB=True,
        show=False
    )

    fig5.savefig(
        subject_dir / "epochs_psd.png",
        dpi=150
    )

    plt.close(fig5)

    # ============================================================
    # 6. Peak-to-Peak Histogram
    # ============================================================

    data = epochs.get_data()

    ptp_per_channel = np.ptp(
        data,
        axis=2
    )

    ptp_per_epoch = ptp_per_channel.max(
        axis=1
    )

    ptp_uv = ptp_per_epoch * 1e6

    fig6 = plt.figure(figsize=(8, 5))

    plt.hist(
        ptp_uv,
        bins=20
    )
    plt.axvline(
        x=250,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Rejection Threshold (250 µV)'
    )
    plt.legend()

    plt.xlabel("Peak-to-Peak Amplitude (µV)")
    plt.ylabel("Epoch Count")
    plt.title(f"{subject} - Peak-to-Peak Distribution")

    plt.tight_layout()

    fig6.savefig(
        subject_dir / "peak_to_peak_histogram.png",
        dpi=150
    )

    plt.close(fig6)

    # ============================================================
    # 7. Statistics CSV
    # ============================================================

    stats = pd.DataFrame(
        {
            "metric": [
                "epochs",
                "mean_ptp_uv",
                "median_ptp_uv",
                "max_ptp_uv",
                "min_ptp_uv",
            ],
            "value": [
                len(ptp_uv),
                ptp_uv.mean(),
                np.median(ptp_uv),
                ptp_uv.max(),
                ptp_uv.min(),
            ],
        }
    )

    stats.to_csv(
        subject_dir / "ptp_stats.csv",
        index=False
    )

    summary_rows.append(
        {
            "subject": subject,
            "epochs": len(ptp_uv),
            "mean_ptp_uv": ptp_uv.mean(),
            "median_ptp_uv": np.median(ptp_uv),
            "max_ptp_uv": ptp_uv.max(),
            "min_ptp_uv": ptp_uv.min(),
        }
    )

    print(
        f"Mean PTP: {ptp_uv.mean():.2f} µV | "
        f"Max PTP: {ptp_uv.max():.2f} µV"
    )

# ============================================================
# Overall Summary
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "all_subjects_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("QC COMPLETE")
print("=" * 60)
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Summary CSV: {OUTPUT_DIR/'all_subjects_summary.csv'}")