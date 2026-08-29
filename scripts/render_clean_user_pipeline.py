"""Render Clean, Beautiful System Pipeline Diagram matching the 4-Stage Container layout.

Fixes all text collisions, coordinate overlaps, and updates parameters:
- Epoch Rejection: ±100 µV (matching configs/preprocessing.yaml)
- Node Features: X in R^{64 x 20} (20-D Spectral Powers)
- Edge Adjacency: Alpha wPLI Top-20% Sparse (A in R^{64 x 64})
- Includes Independent MI Target Generation Branch (CSP+LDA)
- verified benchmark metrics (r = 0.3313, rho = 0.3247, R^2 = +0.1097, MAE = 0.1138)
"""

import os, shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color palette definition
c_blue_bg, c_blue_border = '#E3F2FD', '#0D47A1'
c_green_bg, c_green_border = '#E8F5E9', '#1B5E20'
c_purple_bg, c_purple_border = '#EDE7F6', '#4A148C'
c_orange_bg, c_orange_border = '#FFF3E0', '#E65100'

fig, ax = plt.subplots(figsize=(22, 9.5), dpi=300)
ax.set_xlim(0, 22)
ax.set_ylim(0, 9.5)
ax.axis('off')

def draw_box(ax, x, y, w, h, bg_color, border_color, text, title="", title_fs=8.5, body_fs=7.2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        edgecolor=border_color,
        facecolor=bg_color,
        lw=1.5
    )
    ax.add_patch(rect)

    if title:
        ax.text(
            x + w/2.0,
            y + h - 0.25,
            title,
            fontsize=title_fs,
            fontweight='bold',
            ha='center',
            va='top',
            color=border_color
        )

    body_y = y + (h/2.0 - 0.15) if title else y + h/2.0
    ax.text(
        x + w/2.0,
        body_y,
        text,
        fontsize=body_fs,
        ha='center',
        va='center',
        color='#212121',
        multialignment='center',
        linespacing=1.35
    )

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='-|>',
            color='#374151',
            lw=1.8,
            mutation_scale=14
        )
    )

# ============================================================
# --- Stage Containers ---
# ============================================================

# 1. Preprocessing (x: 0.4 to 5.2)
ax.add_patch(
    patches.FancyBboxPatch(
        (0.4, 0.4), 4.8, 8.3,
        boxstyle="round,pad=0.1",
        facecolor='#F8FAFC',
        edgecolor=c_blue_border,
        ls='--',
        lw=1.5
    )
)
ax.text(2.8, 8.35, "1. Preprocessing Pipeline", fontsize=11, fontweight='bold', color=c_blue_border, ha='center', va='center')


# 2. Graph Construction (x: 5.5 to 10.7)
ax.add_patch(
    patches.FancyBboxPatch(
        (5.5, 0.4), 5.2, 8.3,
        boxstyle="round,pad=0.1",
        facecolor='#F8FAFC',
        edgecolor=c_green_border,
        ls='--',
        lw=1.5
    )
)
ax.text(8.1, 8.35, "2. Graph Construction", fontsize=11, fontweight='bold', color=c_green_border, ha='center', va='center')


# 3. Deep Architecture (x: 11.0 to 15.6)
ax.add_patch(
    patches.FancyBboxPatch(
        (11.0, 0.4), 4.6, 8.3,
        boxstyle="round,pad=0.1",
        facecolor='#F8FAFC',
        edgecolor=c_purple_border,
        ls='--',
        lw=1.5
    )
)
ax.text(13.3, 8.35, "3. Deep GCN Architecture", fontsize=11, fontweight='bold', color=c_purple_border, ha='center', va='center')


# 4. Evaluation & Target (x: 15.9 to 21.6)
ax.add_patch(
    patches.FancyBboxPatch(
        (15.9, 0.4), 5.7, 8.3,
        boxstyle="round,pad=0.1",
        facecolor='#F8FAFC',
        edgecolor=c_orange_border,
        ls='--',
        lw=1.5
    )
)
ax.text(18.75, 8.35, "4. Evaluation & Target", fontsize=11, fontweight='bold', color=c_orange_border, ha='center', va='center')


# ============================================================
# --- Inner Blocks ---
# ============================================================

# Raw Resting EEG
draw_box(
    ax,
    0.7, 5.4, 4.2, 2.4,
    c_blue_bg, c_blue_border,
    "PhysioNet EEGMMIDB (109 Subjects)\n"
    "64 Electrodes, 160 Hz Original Rate\n"
    "R01 (Eyes-Open) & R02 (Eyes-Closed)\n"
    "Audited Defect Handling (S088,S089,S092,S100)",
    "Raw Resting EEG"
)

# Preprocessing
draw_box(
    ax,
    0.7, 0.7, 4.2, 4.2,
    c_blue_bg, c_blue_border,
    "• Resample to 128 Hz (Anti-Aliasing)\n"
    "• Bandpass Filter (1–40 Hz FIR)\n"
    "• Notch Filter (60 Hz Mains Noise)\n"
    "• Common Average Reference (CAR)\n"
    "• Infomax ICA + ICLabel Artifact Removal\n"
    "• 2.0s Fixed Epoching (256 samples)\n"
    "• Peak Rejection (±100 µV, 98.4% Clean)",
    "EEG Preprocessing [FROZEN]"
)


# Node Features
draw_box(
    ax,
    5.8, 5.4, 2.2, 2.4,
    c_green_bg, c_green_border,
    "Welch PSD Band Power\n"
    "(δ, θ, α, β, γ Bands)\n"
    "R01 + R02 Relative &\n"
    "Log-Abs Powers\n"
    "X ∈ R^{64 × 20}",
    "Node Features (X)"
)

# Edge Adjacency
draw_box(
    ax,
    8.2, 5.4, 2.2, 2.4,
    c_green_bg, c_green_border,
    "Alpha-band wPLI\n"
    "(8–13 Hz Phase Lag)\n"
    "Eliminates Zero-Lag\n"
    "Top-20% Sparse Edges\n"
    "A ∈ R^{64 × 64}",
    "Edge Adjacency (A)"
)

# Subject Graph
draw_box(
    ax,
    6.4, 0.7, 3.4, 4.2,
    '#C8E6C9', c_green_border,
    "G_s = (V, E, A, X)\n\n"
    "• 64 Scalp Nodes (V)\n"
    "• 20-D Spectral Attributes (X)\n"
    "• ~403 Sparse wPLI Edges (E)\n"
    "• Symmetric Degree Matrix:\n"
    "  Â = D̃^{-1/2} Ã D̃^{-1/2}",
    "Subject Graph"
)


# 3-Layer GCN
draw_box(
    ax,
    11.3, 0.7, 2.2, 7.1,
    c_purple_bg, c_purple_border,
    "GCN Conv Layer 1 (64 ch)\n+ ReLU Activation\n\n"
    "Dropout (p = 0.2)\n\n"
    "GCN Conv Layer 2 (64 ch)\n+ ReLU Activation\n\n"
    "GCN Conv Layer 3 (64 ch)\n+ ReLU Activation\n\n"
    "Variance-Matched MSE Loss:\n"
    "L = MSE(ŷ, y) + 0.5|Var(ŷ)-Var(y)|",
    "3-Layer GCN Model"
)

# Readout Layer
draw_box(
    ax,
    13.8, 1.5, 1.5, 5.5,
    c_purple_bg, c_purple_border,
    "Global Mean\nPooling\n(h_G ∈ R^64)\n\n"
    "+\n\n"
    "Linear\nRegression\nHead\n(W_r h_G + b_r)",
    "Readout Layer"
)


# Independent Target
draw_box(
    ax,
    16.2, 5.4, 5.1, 2.4,
    '#FFE0B2', c_orange_border,
    "MI Task Runs (R04–R14: Left/Right Fist & Feet Imagery)\n"
    "5-Fold CSP + LDA Classifier (Leakage-Controlled)\n"
    "Continuous Target y_s ∈ [0.25, 1.00] (Mean ȳ = 0.5841)",
    "Independent Target Generation"
)

# Validation Protocol
draw_box(
    ax,
    16.2, 0.7, 2.4, 4.2,
    c_orange_bg, c_orange_border,
    "109-Subject\n"
    "Leave-One-Subject-Out\n"
    "(LOSO Cross-Validation)\n\n"
    "Train on 108 subjects\n"
    "Test on 1 held-out subject\n\n"
    "1000 Label Permutation\n"
    "Test (p = 0.000999)",
    "Validation Protocol"
)

# Target Output
draw_box(
    ax,
    18.9, 0.7, 2.4, 4.2,
    '#FFE0B2', c_orange_border,
    "Predicted MI Performance (ŷ)\n\n"
    "Verified Metrics (N=109):\n"
    "• Pearson r = 0.3313 (p=0.0004)\n"
    "• Spearman ρ = 0.3247 (p=0.0006)\n"
    "• R² Score = +0.1097 (10.97% Var)\n"
    "• MAE = 0.1138 (Lowest error)\n"
    "• RMSE = 0.1456",
    "Target Output & Results"
)


# ============================================================
# --- Arrow Connections ---
# ============================================================

# Raw EEG -> Preprocessing
draw_arrow(ax, 2.8, 5.4, 2.8, 4.9)

# Preprocessing -> Node Features & Edge Adjacency
draw_arrow(ax, 4.9, 2.8, 5.8, 6.6)
draw_arrow(ax, 4.9, 2.8, 8.2, 6.6)

# Node Features & Edge Adjacency -> Subject Graph
draw_arrow(ax, 6.9, 5.4, 7.3, 4.9)
draw_arrow(ax, 9.3, 5.4, 8.7, 4.9)

# Subject Graph -> GCN
draw_arrow(ax, 9.8, 2.8, 11.3, 4.25)

# GCN -> Readout
draw_arrow(ax, 13.5, 4.25, 13.8, 4.25)

# Readout -> Validation Protocol
draw_arrow(ax, 15.3, 4.25, 16.2, 2.8)

# Independent Target -> Target Output
draw_arrow(ax, 18.75, 5.4, 18.75, 4.9)

# Validation -> Target Output
draw_arrow(ax, 18.6, 2.8, 18.9, 2.8)

# ============================================================
# --- Save to Locations ---
# ============================================================

plt.tight_layout(pad=0.8)

p_root = 'system_pipeline.png'
plt.savefig(p_root, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

d1 = r'C:\Users\Admin\OneDrive\Desktop\pdf_figures\system_pipeline.png'
d2 = r'C:\Users\Admin\Desktop\pdf_figures\system_pipeline.png'
d3 = r'C:\Users\Admin\Desktop\publications\figures\system_pipeline.png'

shutil.copy(p_root, d1)
shutil.copy(p_root, d2)
shutil.copy(p_root, d3)

print('Fixed user diagram script and saved clean 300 DPI system_pipeline.png successfully!')
