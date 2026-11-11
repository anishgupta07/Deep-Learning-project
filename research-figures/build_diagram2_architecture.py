"""
GenomeDL v3 - Publication Figure 2: Architecture Vertical
IEEE Style: Detailed Micro-Architecture, Cross-Modal Fusion & Dual Residual Blocks
Outputs: Vector PDF, SVG, 600 DPI PNG
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette (Classic IEEE)
C_TAB_BG     = '#E1F5FE'  # Soft Blue
C_TAB_LINE   = '#0288D1'  # Sky Blue
C_CAT_BG     = '#FFF8E1'  # Soft Amber
C_CAT_LINE   = '#F57F17'  # Amber
C_ESM_BG     = '#EDE7F6'  # Soft Lavender
C_ESM_LINE   = '#512DA8'  # Deep Purple
C_FUSION_BG  = '#FFF3E0'  # Soft Orange
C_FUSION_LINE= '#E65100'  # Deep Orange
C_RES_BG     = '#F3E5F5'  # Soft Magenta
C_RES_LINE   = '#7B1FA2'  # Purple
C_HEAD_BG    = '#E8F5E9'  # Soft Green
C_HEAD_LINE  = '#2E7D32'  # Forest Green
CLR_TEXT     = '#1A1A1A'
CLR_MUTED    = '#424242'

def draw_box(ax, x, y, w, h, bg, border, title, lines=None, tag=None, rx=0.06, title_color=CLR_TEXT):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                         facecolor=bg, edgecolor=border, linewidth=1.1, zorder=2)
    ax.add_patch(box)
    
    if lines:
        ax.text(x + w/2, y + h - 0.20, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color=title_color, zorder=3)
        y_text = y + h - 0.40
        for l in lines:
            ax.text(x + w/2, y_text, l, ha='center', va='center', fontsize=7.2, color=CLR_MUTED, zorder=3)
            y_text -= 0.18
        if tag:
            ax.text(x + w/2, y + 0.15, tag, ha='center', va='center', fontsize=7.4, fontweight='bold', color=border, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color=title_color, zorder=3)
    return box

def draw_arrow(ax, p1, p2, color='#455A64', lw=1.1, label=None, label_pos=(0, 0), curved=0):
    connectionstyle = f"arc3,rad={curved}" if curved != 0 else "arc3"
    arrow = patches.FancyArrowPatch(p1, p2, connectionstyle=connectionstyle,
                                    arrowstyle='-|>', mutation_scale=9, linewidth=lw, color=color, zorder=4)
    ax.add_patch(arrow)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_pos[0]
        my = (p1[1] + p2[1]) / 2 + label_pos[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=7.0, color=color, fontweight='medium', zorder=5)

def build_architecture_diagram(out_dir):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Figure Header
    ax.text(8.0, 9.75, "FIGURE 2: FusionResNet Micro-Architecture & Tensor Transformation Flow",
            ha='center', va='center', fontsize=13.0, fontweight='bold', color='#111111')
    ax.text(8.0, 9.48, "Deep residual neural network featuring isolated branch projection, balanced latent fusion, dual ResBlocks, and calibrated output",
            ha='center', va='center', fontsize=8.6, style='italic', color='#555555')

    # -------------------------------------------------------------
    # 3 INPUT BRANCHES (TOP LEVEL)
    # -------------------------------------------------------------
    # 1. Tabular Branch
    draw_box(ax, 0.5, 7.8, 3.8, 1.45, C_TAB_BG, C_TAB_LINE, "Tabular Branch (Dense Projection)",
             [r"Input: $\mathbf{x}_{\mathrm{tab}} \in \mathbb{R}^{26}$ (7 Continuous + 19 Binary)",
              r"$\mathbf{z}_{\mathrm{tab}} = \mathrm{Linear}(26 \to 64)(\mathbf{x}_{\mathrm{tab}})$",
              r"$\mathbf{h}_{\mathrm{tab}} = \mathrm{GELU}(\mathrm{LayerNorm}(\mathbf{z}_{\mathrm{tab}}))$"],
             tag=r"Output: $\mathbf{h}_{\mathrm{tab}} \in \mathbb{R}^{64}$", rx=0.06)

    # 2. Categorical Branch
    draw_box(ax, 5.0, 7.8, 4.3, 1.45, C_CAT_BG, C_CAT_LINE, "Categorical Embeddings Branch",
             [r"Inputs: $\mathbf{c}_{\mathrm{class}} \in \{0..11\}, \mathbf{c}_{\mathrm{synd}} \in \{0..14\}$",
              r"$\mathbf{e}_{\mathrm{class}} = \mathrm{Embedding}(12 \to 8)(\mathbf{c}_{\mathrm{class}})$",
              r"$\mathbf{e}_{\mathrm{synd}} = \mathrm{Embedding}(15 \to 8)(\mathbf{c}_{\mathrm{synd}})$",
              r"$\mathbf{h}_{\mathrm{cat}} = [\mathbf{e}_{\mathrm{class}} \,\|\, \mathbf{e}_{\mathrm{synd}}]$"],
             tag=r"Output: $\mathbf{h}_{\mathrm{cat}} \in \mathbb{R}^{16}$", rx=0.06)

    # 3. ESM Branch
    draw_box(ax, 9.9, 7.8, 5.5, 1.45, C_ESM_BG, C_ESM_LINE, "ESM-2 Protein Representation Branch",
             [r"Inputs: $\mathrm{LLR} \in \mathbb{R}, \mathrm{Avail} \in \{0, 1\}, \mathbf{e}_{\mathrm{mask}} \in \mathbb{R}^{480}$",
              r"$\mathbf{p}_{\mathrm{esm}} = \mathrm{Linear}(480 \to 64)(\mathbf{e}_{\mathrm{mask}})$",
              r"$\mathbf{x}_{\mathrm{esm\_in}} = [\mathrm{LLR} \,\|\, \mathrm{Avail} \,\|\, \mathbf{p}_{\mathrm{esm}}] \in \mathbb{R}^{66}$",
              r"$\mathbf{h}_{\mathrm{esm}} = \mathrm{Linear}(64 \to 64)(\mathrm{GELU}(\mathrm{LN}(\mathrm{Linear}(66 \to 64)(\mathbf{x}_{\mathrm{esm\_in}}))))$"],
             tag=r"Output: $\mathbf{h}_{\mathrm{esm}} \in \mathbb{R}^{64}$", rx=0.06)

    # -------------------------------------------------------------
    # MULTIMODAL CONCATENATION & PROJECTION
    # -------------------------------------------------------------
    draw_arrow(ax, (2.4, 7.8), (6.5, 6.7), color=C_TAB_LINE, lw=1.2)
    draw_arrow(ax, (7.15, 7.8), (7.5, 6.7), color=C_CAT_LINE, lw=1.2)
    draw_arrow(ax, (12.65, 7.8), (8.5, 6.7), color=C_ESM_LINE, lw=1.2)

    draw_box(ax, 4.2, 5.75, 6.6, 0.95, C_FUSION_BG, C_FUSION_LINE, "Multimodal Fusion Layer",
             [r"$\mathbf{h}_{\mathrm{fused}} = [\mathbf{h}_{\mathrm{tab}} \,\|\, \mathbf{h}_{\mathrm{cat}} \,\|\, \mathbf{h}_{\mathrm{esm}}] \in \mathbb{R}^{64 + 16 + 64 = 144}$",
              r"$\mathbf{x}_{0} = \mathrm{GELU}(\mathrm{LayerNorm}(\mathrm{Linear}(144 \to 256)(\mathbf{h}_{\mathrm{fused}})))$"],
             tag=r"Latent Manifold: $\mathbf{x}_{0} \in \mathbb{R}^{256}$", rx=0.06)

    draw_arrow(ax, (7.5, 5.75), (7.5, 5.15), color=C_FUSION_LINE, lw=1.4)

    # -------------------------------------------------------------
    # RESIDUAL BLOCKS 1 & 2
    # -------------------------------------------------------------
    # Detailed ResBlock 1 Box
    draw_box(ax, 3.2, 3.25, 8.6, 1.9, C_RES_BG, C_RES_LINE, "Residual Block 1 (Dimension = 256)",
             [r"Identity Path: $\mathbf{x}_{0} \in \mathbb{R}^{256}$",
              r"Transform Path $\mathcal{F}(\mathbf{x}_{0})$:",
              r"  $\mathbf{z}_{1} = \mathrm{Linear}(256 \to 256)(\mathrm{Dropout}_{0.2}(\mathrm{GELU}(\mathrm{LayerNorm}(\mathbf{x}_{0}))))$",
              r"  $\mathbf{z}_{2} = \mathrm{Linear}(256 \to 256)(\mathrm{Dropout}_{0.2}(\mathrm{GELU}(\mathrm{LayerNorm}(\mathbf{z}_{1}))))$",
              r"Residual Additive Recombination:",
              r"  $\mathbf{x}_{1} = \mathbf{x}_{0} + \mathbf{z}_{2} \in \mathbb{R}^{256}$"],
             tag=r"$\mathbf{x}_{1} = \mathbf{x}_{0} + \mathcal{F}(\mathbf{x}_{0})$", rx=0.07)

    # Skip arrow visual
    ax.annotate("", xy=(3.2, 3.65), xytext=(3.2, 4.75),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.8", color='#6A1B9A', lw=1.3, linestyle='--'))
    ax.text(2.1, 4.2, r"Identity Skip $\mathbf{x}_{0}$", ha='center', va='center', fontsize=7.2, color='#6A1B9A', fontweight='bold')

    draw_arrow(ax, (7.5, 3.25), (7.5, 2.70), color=C_RES_LINE, lw=1.4)

    # Detailed ResBlock 2 Box
    draw_box(ax, 3.2, 0.80, 8.6, 1.9, C_RES_BG, C_RES_LINE, "Residual Block 2 (Dimension = 256)",
             [r"Identity Path: $\mathbf{x}_{1} \in \mathbb{R}^{256}$",
              r"Transform Path $\mathcal{F}(\mathbf{x}_{1})$:",
              r"  $\mathbf{z}_{3} = \mathrm{Linear}(256 \to 256)(\mathrm{Dropout}_{0.2}(\mathrm{GELU}(\mathrm{LayerNorm}(\mathbf{x}_{1}))))$",
              r"  $\mathbf{z}_{4} = \mathrm{Linear}(256 \to 256)(\mathrm{Dropout}_{0.2}(\mathrm{GELU}(\mathrm{LayerNorm}(\mathbf{z}_{3}))))$",
              r"Residual Additive Recombination:",
              r"  $\mathbf{x}_{2} = \mathbf{x}_{1} + \mathbf{z}_{4} \in \mathbb{R}^{256}$"],
             tag=r"$\mathbf{x}_{2} = \mathbf{x}_{1} + \mathcal{F}(\mathbf{x}_{1})$", rx=0.07)

    # Skip arrow visual 2
    ax.annotate("", xy=(3.2, 1.2), xytext=(3.2, 2.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.8", color='#6A1B9A', lw=1.3, linestyle='--'))
    ax.text(2.1, 1.75, r"Identity Skip $\mathbf{x}_{1}$", ha='center', va='center', fontsize=7.2, color='#6A1B9A', fontweight='bold')

    # -------------------------------------------------------------
    # PREDICTION HEAD & CALIBRATION (RIGHT SIDE)
    # -------------------------------------------------------------
    draw_arrow(ax, (11.8, 1.75), (12.5, 1.75), color=C_RES_LINE, lw=1.4)

    draw_box(ax, 12.5, 0.65, 3.1, 2.2, C_HEAD_BG, C_HEAD_LINE, "Prediction & Calibration Head",
             [r"1. Linear Head:",
              r"   $z = \mathrm{Linear}(256 \to 1)(\mathrm{GELU}(\mathrm{LN}(\mathbf{x}_{2})))$",
              r"2. Platt Calibration (Validation L-BFGS):",
              r"   $z_{\mathrm{cal}} = s \cdot z + b$",
              r"   ($s = 1.0559, b = -1.5561$)",
              r"3. Sigmoid Activation:",
              r"   $P(\mathrm{Pathogenic}) = \sigma(z_{\mathrm{cal}})$",
              r"   $\mathrm{ECE}: 0.0279 \to 0.0009$"],
             tag=r"Calibrated $P \in [0.0, 1.0]$", rx=0.06)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "figure2_architecture_micro_details.pdf")
    svg_path = os.path.join(out_dir, "figure2_architecture_micro_details.svg")
    png_path = os.path.join(out_dir, "figure2_architecture_micro_details.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    fig.savefig(svg_path, format="svg", bbox_inches='tight')
    fig.savefig(png_path, format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 2 (Architecture Micro Details) generated:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\monoj\OneDrive\Desktop\deepL\Deep-Learning-project\research-figures"
    build_architecture_diagram(out_dir)
