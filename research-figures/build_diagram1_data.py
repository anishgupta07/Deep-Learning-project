"""
GenomeDL v3 - Publication Figure 1: Data Vertical
IEEE Style: Multimodal Feature Engineering & Cross-Domain Representation Pipeline
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
C_RAW_BG      = '#ECEFF1'  # Light Slate Grey
C_RAW_LINE    = '#455A64'  # Deep Slate
C_RESOLVE_BG  = '#E0F7FA'  # Cyan / Light Teal
C_RESOLVE_LINE= '#00838F'  # Dark Teal
C_CHEM_BG     = '#E1F5FE'  # Soft Blue
C_CHEM_LINE   = '#0288D1'  # Deep Sky Blue
C_CAT_BG      = '#FFF8E1'  # Soft Amber
C_CAT_LINE    = '#F57F17'  # Deep Amber
C_ESM_BG      = '#EDE7F6'  # Soft Lavender
C_ESM_LINE    = '#512DA8'  # Deep Purple
C_OUT_BG      = '#E8F5E9'  # Soft Green
C_OUT_LINE    = '#2E7D32'  # Forest Green
CLR_TEXT      = '#1A1A1A'
CLR_MUTED     = '#424242'

def draw_box(ax, x, y, w, h, bg, border, title, lines=None, tag=None, rx=0.06, title_color=CLR_TEXT):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                         facecolor=bg, edgecolor=border, linewidth=1.1, zorder=2)
    ax.add_patch(box)
    
    if lines:
        ax.text(x + w/2, y + h - 0.22, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color=title_color, zorder=3)
        y_text = y + h - 0.44
        for l in lines:
            ax.text(x + w/2, y_text, l, ha='center', va='center', fontsize=7.1, color=CLR_MUTED, zorder=3)
            y_text -= 0.20
        if tag:
            ax.text(x + w/2, y + 0.16, tag, ha='center', va='center', fontsize=7.4, fontweight='bold', color=border, zorder=3)
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

def build_data_pipeline_diagram(out_dir):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Figure Header
    ax.text(8.0, 9.75, "FIGURE 1: Multimodal Feature Engineering & Cross-Domain Representation Pipeline",
            ha='center', va='center', fontsize=13.0, fontweight='bold', color='#111111')
    ax.text(8.0, 9.48, "Automated extraction translating raw genomic coordinates into biophysical, categorical, and protein language model tensors",
            ha='center', va='center', fontsize=8.6, style='italic', color='#555555')

    # STAGE 1: RAW INPUT & COORDINATE RESOLUTION
    draw_box(ax, 0.4, 7.5, 2.6, 1.5, C_RAW_BG, C_RAW_LINE, "Raw Genomic Variant",
             ["Chromosome, Position (GRCh38)", "Reference Allele (Ref)", "Alternate Allele (Alt)", "HGVS / Gene Identifier"],
             tag="Input: VCF / TSV", rx=0.08)

    draw_arrow(ax, (3.0, 8.25), (3.6, 8.25), color=C_RAW_LINE, lw=1.3)

    draw_box(ax, 3.6, 7.3, 3.3, 1.9, C_RESOLVE_BG, C_RESOLVE_LINE, "Coordinate & Transcript Engine",
             ["Query panel.json (130 Panel Genes)",
              "Identify Canonical MANE Transcript",
              "Resolve Strand Orientation (+ / -)",
              "Map Relative Protein Position:",
              r"$p_{\mathrm{rel}} = \mathrm{pos}_{\mathrm{AA}} / L_{\mathrm{protein}} \in [0, 1]$",
              "Coding-Strand Complement if Minus"],
             tag="Transcript & Coding Frame Alignment", rx=0.08)

    # 3 Branches Split Point
    draw_arrow(ax, (6.9, 8.25), (7.5, 8.25), color=C_RESOLVE_LINE, lw=1.4)
    ax.plot([7.5, 7.5], [2.15, 8.25], color=C_RESOLVE_LINE, lw=1.3, linestyle='-', zorder=4)

    # -------------------------------------------------------------
    # BRANCH A: CHEMINFORMATICS & MOLECULAR BIOPHYSICS
    # -------------------------------------------------------------
    draw_arrow(ax, (7.5, 8.25), (8.0, 8.25), color=C_CHEM_LINE, lw=1.2)
    draw_box(ax, 8.0, 7.0, 4.3, 2.3, C_CHEM_BG, C_CHEM_LINE, "Branch A: Cheminformatics & Stereochemistry (RDKit)",
             ["1. Amino Acid Sidechain SMILES Parsing",
              r"2. Grantham Physicochemical Distance Matrix (Grantham 1974)",
              r"3. Thermodynamic & Charge Deltas: $\Delta\mathrm{MW}, \Delta\mathrm{Charge}$",
              r"4. Solvation Partition & Polar Surface Area: $\Delta\mathrm{LogP}, \Delta\mathrm{TPSA}$",
              r"5. Morgan Fingerprint Tanimoto Distance ($d_{\mathrm{T}} = 1 - S_{\mathrm{Tanimoto}}$)",
              r"6. Genomic Context: 8 Coding-Strand One-Hots, Transitions, Isoform"],
             tag="7 Continuous Features + 19 Binary Indicators", rx=0.06)

    draw_arrow(ax, (12.3, 8.15), (13.0, 8.15), color=C_CHEM_LINE)
    draw_box(ax, 13.0, 7.45, 2.5, 1.4, C_CHEM_BG, C_CHEM_LINE, "Tabular Processing",
             ["StandardScaler fit on Train:",
              r"$z = (x_{\mathrm{cont}} - \mu) / \sigma$",
              r"Stacking: $[z; x_{\mathrm{bin}}]$"],
             tag=r"$\mathbf{x}_{\mathrm{tab}} \in \mathbb{R}^{26}$", rx=0.06)

    # -------------------------------------------------------------
    # BRANCH B: CATEGORICAL SYNDROME & CONSEQUENCE TAXONOMY
    # -------------------------------------------------------------
    draw_arrow(ax, (7.5, 5.2), (8.0, 5.2), color=C_CAT_LINE, lw=1.2)
    draw_box(ax, 8.0, 4.2, 4.3, 2.0, C_CAT_BG, C_CAT_LINE, "Branch B: Categorical Taxonomy Classification",
             ["Variant Consequence Parsing (12 Classes):",
              "  synonymous, missense, nonsense, start_lost, stop_lost,",
              "  splice_canonical, splice_region, intronic, utr5, utr3...",
              "Syndrome Group Association (15 Groups):",
              "  Breast/Ovarian HR, Lynch/MMR, Li-Fraumeni, Polyposis..."],
             tag="Discrete Categorical Indices", rx=0.06)

    draw_arrow(ax, (12.3, 5.2), (13.0, 5.2), color=C_CAT_LINE)
    draw_box(ax, 13.0, 4.5, 2.5, 1.4, C_CAT_BG, C_CAT_LINE, "Embedding Indexing",
             ["Variant Class ID: 0 ... 11",
              "Syndrome Group ID: 0 ... 14",
              "Lookups into 8-d Latent Spaces"],
             tag=r"$\mathbf{c}_{\mathrm{class}}, \mathbf{c}_{\mathrm{synd}} \in \mathbb{N}$", rx=0.06)

    # -------------------------------------------------------------
    # BRANCH C: EVOLUTIONARY PROTEIN LANGUAGE MODEL (ESM-2)
    # -------------------------------------------------------------
    draw_arrow(ax, (7.5, 2.15), (8.0, 2.15), color=C_ESM_LINE, lw=1.2)
    draw_box(ax, 8.0, 0.95, 4.3, 2.4, C_ESM_BG, C_ESM_LINE, "Branch C: Protein Language Modeling (ESM-2 t12_35M)",
             ["1. Query Canonical Protein Sequence (panel_proteins.fasta)",
              "2. Context Window Centering on Mutation Site (max length 1022 AA)",
              "3. Mask Mutation Residue: M-E-E-P-Q-[MASK]-D-P-S-V...",
              r"4. Masked-Marginal Scoring: $\mathrm{LLR} = \log P(\mathrm{AA}_{\mathrm{alt}} | x_{\backslash i}) - \log P(\mathrm{AA}_{\mathrm{ref}} | x_{\backslash i})$",
              "5. Layer-12 Transformer Hidden State Extraction at Mask Position",
              "6. Residue Availability Flag: 1.0 (protein) / 0.0 (non-coding)"],
             tag="Evolutionary Plausibility & 3D Spatial Context", rx=0.06)

    draw_arrow(ax, (12.3, 2.15), (13.0, 2.15), color=C_ESM_LINE)
    draw_box(ax, 13.0, 1.35, 2.5, 1.6, C_ESM_BG, C_ESM_LINE, "ESM Tensor Packaging",
             [r"Scalar $\mathrm{LLR} \in \mathbb{R}$",
              r"Availability Flag $\in \{0, 1\}$",
              r"Residue State: $\mathbf{e}_{\mathrm{mask}} \in \mathbb{R}^{480}$",
              "Caches into Parquet/PyTorch"],
             tag=r"$\mathbf{x}_{\mathrm{esm}} = [\mathrm{LLR}, \mathrm{Avail}, \mathbf{e}_{\mathrm{mask}}]$", rx=0.06)

    # BOTTOM FINAL UNIFIED BATCH TENSOR
    draw_box(ax, 3.5, 0.12, 9.0, 0.65, C_OUT_BG, C_OUT_LINE,
             "Unified Multimodal Dataset Sample",
             [r"Dataloader Item = { 'tabular': $\mathbf{x}_{\mathrm{tab}} [26]$, 'vclass': $\mathbf{c}_{\mathrm{class}}$, 'syndrome': $\mathbf{c}_{\mathrm{synd}}$, 'esm_llr': $[1]$, 'esm_avail': $[1]$, 'esm_emb': $[480]$ }"],
             tag=r"Tensorized Batch: $[\mathcal{B} \times (26 + 1 + 1 + 482)]$", rx=0.05, title_color='#1B5E20')

    # Routing arrows to unified batch
    draw_arrow(ax, (14.25, 7.45), (14.25, 0.45), color='#0288D1', curved=0.3)
    draw_arrow(ax, (14.25, 4.5), (13.8, 0.45), color='#F57F17', curved=0.2)
    draw_arrow(ax, (13.0, 1.8), (12.5, 0.45), color='#512DA8', curved=0.1)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "figure1_data_feature_pipeline.pdf")
    svg_path = os.path.join(out_dir, "figure1_data_feature_pipeline.svg")
    png_path = os.path.join(out_dir, "figure1_data_feature_pipeline.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    fig.savefig(svg_path, format="svg", bbox_inches='tight')
    fig.savefig(png_path, format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 1 (Data Pipeline) generated:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\monoj\OneDrive\Desktop\deepL\Deep-Learning-project\research-figures"
    build_data_pipeline_diagram(out_dir)
