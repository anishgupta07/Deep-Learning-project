"""
GenomeDL v3 - Publication Figure 1 Generator
IEEE Style 2D Complete End-to-End Multimodal Architecture Diagram
Vector PDF, SVG, and 600 DPI PNG outputs.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

# Configure IEEE Publication Typography & Styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Standard IEEE Color Palette
CLR_TABULAR   = '#E1F5FE'  # Soft light blue (tabular & features)
CLR_TAB_LINE  = '#0288D1'  # Rich blue border
CLR_EMBED     = '#FFF8E1'  # Soft amber/yellow (discrete embeddings)
CLR_EMB_LINE  = '#F57F17'  # Amber border
CLR_ESM       = '#EDE7F6'  # Soft purple/indigo (protein LM branch)
CLR_ESM_LINE  = '#512DA8'  # Deep purple border
CLR_NORM_ACT  = '#E0F2F1'  # Soft teal (normalization & activation)
CLR_NA_LINE   = '#00796B'  # Teal border
CLR_FUSION    = '#FFF3E0'  # Soft orange (multimodal concatenation & fusion)
CLR_FUS_LINE  = '#E65100'  # Deep orange border
CLR_RESBLOCK  = '#F3E5F5'  # Soft magenta (residual blocks)
CLR_RES_LINE  = '#7B1FA2'  # Purple/magenta border
CLR_OUTPUT    = '#E8F5E9'  # Soft green (prediction head & calibration)
CLR_OUT_LINE  = '#2E7D32'  # Forest green border
CLR_TEXT      = '#1A1A1A'  # Near black for crisp typography
CLR_MUTED     = '#555555'  # Muted grey for sub-labels

def draw_box(ax, x, y, w, h, bg_color, border_color, title, subtitle=None, shape_tag=None, rx=0.08):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                         facecolor=bg_color, edgecolor=border_color, linewidth=1.1, zorder=2)
    ax.add_patch(box)
    
    if subtitle and shape_tag:
        ax.text(x + w/2, y + h*0.72, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=CLR_TEXT, zorder=3)
        ax.text(x + w/2, y + h*0.48, subtitle, ha='center', va='center', fontsize=7.2, color=CLR_MUTED, zorder=3)
        ax.text(x + w/2, y + h*0.22, shape_tag, ha='center', va='center', fontsize=7.5, fontweight='bold', color=border_color, zorder=3)
    elif subtitle:
        ax.text(x + w/2, y + h*0.64, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=CLR_TEXT, zorder=3)
        ax.text(x + w/2, y + h*0.34, subtitle, ha='center', va='center', fontsize=7.5, color=CLR_MUTED, zorder=3)
    elif shape_tag:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=CLR_TEXT, zorder=3)
        ax.text(x + w/2, y + h*0.32, shape_tag, ha='center', va='center', fontsize=7.5, fontweight='bold', color=border_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=CLR_TEXT, zorder=3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, label=None, color='#444444', lw=1.1, label_side='right', curved=0):
    if curved != 0:
        connectionstyle = f"arc3,rad={curved}"
    else:
        connectionstyle = "arc3"
    
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle=connectionstyle,
                                    arrowstyle='-|>',
                                    mutation_scale=9,
                                    linewidth=lw,
                                    color=color,
                                    zorder=4)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        offset_x = 0.12 if label_side == 'right' else -0.12
        ha = 'left' if label_side == 'right' else 'right'
        ax.text(mx + offset_x, my, label, ha=ha, va='center', fontsize=7.2, color=color, fontweight='medium', zorder=5)

def build_figure_1(out_dir):
    fig = plt.figure(figsize=(15.5, 9.5), dpi=300)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Main Figure Title Block
    ax.text(8.0, 9.75, "FIGURE 1: Multimodal Fusion ResNet Architecture for Pathogenicity Prediction",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color='#111111')
    ax.text(8.0, 9.48, "Tri-branch deep neural network integrating biochemical tabular descriptors, categorical embeddings, and ESM-2 protein language representations",
            ha='center', va='center', fontsize=8.8, style='italic', color='#444444')

    # Background Category Container Boxes (Dashed lines to demarcate modalities)
    tab_cont = patches.Rectangle((0.3, 4.8), 4.7, 4.45, linewidth=0.8, edgecolor='#B0BEC5', facecolor='#FAFAFA', linestyle='--', zorder=0)
    cat_cont = patches.Rectangle((5.3, 4.8), 4.9, 4.45, linewidth=0.8, edgecolor='#FFE082', facecolor='#FFFDE7', linestyle='--', zorder=0)
    esm_cont = patches.Rectangle((10.5, 4.8), 5.2, 4.45, linewidth=0.8, edgecolor='#D1C4E9', facecolor='#F3E5F5', linestyle='--', zorder=0)
    ax.add_patch(tab_cont)
    ax.add_patch(cat_cont)
    ax.add_patch(esm_cont)

    ax.text(2.65, 9.08, "BRANCH 1: TABULAR BIOCHEMISTRY", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#0277BD')
    ax.text(7.75, 9.08, "BRANCH 2: CATEGORICAL EMBEDDINGS", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#F57F17')
    ax.text(13.1, 9.08, "BRANCH 3: ESM-2 LANGUAGE MODEL", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#512DA8')

    # -------------------------------------------------------------
    # BRANCH 1: TABULAR BIOCHEMISTRY
    # -------------------------------------------------------------
    # Raw Inputs
    draw_box(ax, 0.5, 8.0, 2.0, 0.85, CLR_TABULAR, CLR_TAB_LINE, "Continuous Feats.", "Grantham, ΔMW, ΔCharge,\nΔLogP, ΔTPSA, Tanimoto", "[B, 7]")
    draw_box(ax, 2.8, 8.0, 2.0, 0.85, CLR_TABULAR, CLR_TAB_LINE, "Binary Indicators", "Transitions, Transcripts,\nNucleotide One-Hots", "[B, 19]")
    
    # Preprocessing / StandardScaler
    draw_box(ax, 0.5, 6.75, 2.0, 0.70, CLR_NORM_ACT, CLR_NA_LINE, "StandardScaler", "z = (x - μ) / σ (Train)", "[B, 7]")
    draw_arrow(ax, 1.5, 8.0, 1.5, 7.45)
    
    # Concatenate continuous + binary
    draw_box(ax, 1.5, 5.75, 2.3, 0.65, CLR_FUSION, CLR_FUS_LINE, "Tabular Stacking", "Concat(scaled_cont, binary)", "[B, 26]")
    draw_arrow(ax, 1.5, 6.75, 2.2, 6.4)
    draw_arrow(ax, 3.8, 8.0, 3.0, 6.4)

    # Linear + LayerNorm + GELU
    draw_box(ax, 1.5, 4.95, 2.3, 0.60, CLR_TABULAR, CLR_TAB_LINE, "Dense Layer + LN", "Linear(26→64) + LN + GELU", "h_tab: [B, 64]")
    draw_arrow(ax, 2.65, 5.75, 2.65, 5.55)

    # -------------------------------------------------------------
    # BRANCH 2: CATEGORICAL EMBEDDINGS
    # -------------------------------------------------------------
    draw_box(ax, 5.5, 8.0, 2.1, 0.85, CLR_EMBED, CLR_EMB_LINE, "Variant Class ID", "12 discrete classes\n(missense, nonsense, ...)", "[B, 1]")
    draw_box(ax, 7.9, 8.0, 2.1, 0.85, CLR_EMBED, CLR_EMB_LINE, "Syndrome Group ID", "15 cancer syndromes\n(Lynch, Li-Fraumeni, ...)", "[B, 1]")

    draw_box(ax, 5.5, 6.75, 2.1, 0.70, CLR_EMBED, CLR_EMB_LINE, "Embedding Layer", "vocab=12, dim=8", "[B, 8]")
    draw_box(ax, 7.9, 6.75, 2.1, 0.70, CLR_EMBED, CLR_EMB_LINE, "Embedding Layer", "vocab=15, dim=8", "[B, 8]")
    draw_arrow(ax, 6.55, 8.0, 6.55, 7.45)
    draw_arrow(ax, 8.95, 8.0, 8.95, 7.45)

    draw_box(ax, 6.6, 5.40, 2.3, 0.70, CLR_FUSION, CLR_FUS_LINE, "Concat Embeddings", "Concat(vclass, syndrome)", "h_cat: [B, 16]")
    draw_arrow(ax, 6.55, 6.75, 7.25, 6.10)
    draw_arrow(ax, 8.95, 6.75, 8.25, 6.10)

    # -------------------------------------------------------------
    # BRANCH 3: ESM-2 LANGUAGE MODEL BRANCH
    # -------------------------------------------------------------
    draw_box(ax, 10.7, 8.0, 1.4, 0.85, CLR_ESM, CLR_ESM_LINE, "ESM LLR", "log P(alt|ctx)\n- log P(ref|ctx)", "[B, 1]")
    draw_box(ax, 12.3, 8.0, 1.4, 0.85, CLR_ESM, CLR_ESM_LINE, "ESM Avail", "Indicator\nProtein vs Noncod.", "[B, 1]")
    draw_box(ax, 13.9, 8.0, 1.6, 0.85, CLR_ESM, CLR_ESM_LINE, "ESM Hidden Rep", "t12_35M_UR50D\nMasked Layer-12", "[B, 480]")

    draw_box(ax, 13.9, 6.75, 1.6, 0.70, CLR_ESM, CLR_ESM_LINE, "Linear Projection", "Linear(480→64)", "[B, 64]")
    draw_arrow(ax, 14.7, 8.0, 14.7, 7.45)

    draw_box(ax, 11.5, 5.75, 3.2, 0.65, CLR_FUSION, CLR_FUS_LINE, "Concat ESM Features", "Concat([LLR, Avail, Projected])", "[B, 66]")
    draw_arrow(ax, 11.4, 8.0, 12.3, 6.4)
    draw_arrow(ax, 13.0, 8.0, 13.0, 6.4)
    draw_arrow(ax, 14.7, 6.75, 13.7, 6.4)

    draw_box(ax, 11.5, 4.95, 3.2, 0.60, CLR_ESM, CLR_ESM_LINE, "ESM MLP Subnetwork", "Linear(66→64)+LN+GELU+Linear(64→64)", "h_esm: [B, 64]")
    draw_arrow(ax, 13.1, 5.75, 13.1, 5.55)

    # -------------------------------------------------------------
    # MULTIMODAL CONCATENATION & FUSION
    # -------------------------------------------------------------
    fusion_box = draw_box(ax, 4.8, 3.65, 6.4, 0.70, CLR_FUSION, CLR_FUS_LINE,
                          "Multimodal Fusion Layer", "Concat(h_tab [64], h_cat [16], h_esm [64]) → Linear(144→256) + LN + GELU", "fused: [B, 256]")
    
    # Concat arrows from 3 branches into fusion box
    draw_arrow(ax, 2.65, 4.95, 5.5, 4.35, label="h_tab", label_side='left')
    draw_arrow(ax, 7.75, 5.40, 7.75, 4.35, label="h_cat")
    draw_arrow(ax, 13.1, 4.95, 10.2, 4.35, label="h_esm", label_side='right')

    # -------------------------------------------------------------
    # RESIDUAL BLOCKS (1 & 2)
    # -------------------------------------------------------------
    draw_arrow(ax, 8.0, 3.65, 8.0, 3.15, label="256-d")
    
    res1 = draw_box(ax, 5.2, 2.30, 5.6, 0.85, CLR_RESBLOCK, CLR_RES_LINE,
                    "Residual Block 1 (x + f(x))",
                    "LayerNorm → GELU → Dropout(0.2) → Linear(256→256) →\nLayerNorm → GELU → Dropout(0.2) → Linear(256→256)",
                    "[B, 256]")
    # Skip connection loop 1
    ax.annotate("", xy=(8.0, 2.30), xytext=(8.0, 3.15),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.95", color='#6A1B9A', lw=1.3, linestyle='--'))
    ax.text(10.95, 2.72, "Identity Skip\nx + f(x)", ha='left', va='center', fontsize=7.2, color='#6A1B9A', fontweight='bold')

    draw_arrow(ax, 8.0, 2.30, 8.0, 1.80, label="256-d")

    res2 = draw_box(ax, 5.2, 0.95, 5.6, 0.85, CLR_RESBLOCK, CLR_RES_LINE,
                    "Residual Block 2 (x + f(x))",
                    "LayerNorm → GELU → Dropout(0.2) → Linear(256→256) →\nLayerNorm → GELU → Dropout(0.2) → Linear(256→256)",
                    "[B, 256]")
    # Skip connection loop 2
    ax.annotate("", xy=(8.0, 0.95), xytext=(8.0, 1.80),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.95", color='#6A1B9A', lw=1.3, linestyle='--'))
    ax.text(10.95, 1.37, "Identity Skip\nx + f(x)", ha='left', va='center', fontsize=7.2, color='#6A1B9A', fontweight='bold')

    # -------------------------------------------------------------
    # OUTPUT PREDICTION HEAD & CALIBRATION
    # -------------------------------------------------------------
    draw_arrow(ax, 8.0, 0.95, 8.0, 0.55)
    
    # Dense Head + Platt Scaler
    draw_box(ax, 1.5, 0.05, 4.8, 0.65, CLR_OUTPUT, CLR_OUT_LINE,
             "Prediction Head", "LayerNorm(256) → GELU → Linear(256→1)", "Raw Logit z: [B, 1]")
    
    draw_arrow(ax, 6.3, 0.37, 6.8, 0.37)

    draw_box(ax, 6.8, 0.05, 4.4, 0.65, CLR_OUTPUT, CLR_OUT_LINE,
             "Platt Scaling Calibration", "z_calibrated = s · z + b (s=1.0559, b=-1.5561)\nECE: 0.0279 → 0.0009", "Calibrated Logit: [B, 1]")
    
    draw_arrow(ax, 11.2, 0.37, 11.7, 0.37)

    draw_box(ax, 11.7, 0.05, 3.8, 0.65, CLR_OUTPUT, CLR_OUT_LINE,
             "Sigmoid Activation", r"P(Pathogenic) = $\sigma(z_{cal})$" + "\nBinary Decision Threshold = 0.50", "Probability in [0, 1]")

    # Wire to final output
    draw_arrow(ax, 8.0, 0.95, 3.9, 0.70, curved=-0.2)

    # Save to multiple publication formats
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "figure1_complete_architecture.pdf")
    svg_path = os.path.join(out_dir, "figure1_complete_architecture.svg")
    png_path = os.path.join(out_dir, "figure1_complete_architecture.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    fig.savefig(svg_path, format="svg", bbox_inches='tight')
    fig.savefig(png_path, format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 1 successfully:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\monoj\OneDrive\Desktop\deepL\Deep-Learning-project\research-figures"
    build_figure_1(out_dir)
