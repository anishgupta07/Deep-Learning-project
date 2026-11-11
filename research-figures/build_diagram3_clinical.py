"""
GenomeDL v3 - Publication Figure 3: Clinical Decision & Calibration Vertical
IEEE Style: Hybrid Clinical Decision Pipeline, Platt Calibration & Penetrance Mapping
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
C_INPUT_BG    = '#ECEFF1'  # Light Slate
C_INPUT_LINE  = '#455A64'  # Slate
C_FILTER_BG   = '#FFF9C4'  # Soft Yellow
C_FILTER_LINE = '#FBC02D'  # Dark Gold
C_PASS_BG     = '#E8F5E9'  # Soft Green (Benign)
C_PASS_LINE   = '#2E7D32'  # Forest Green
C_WARN_BG     = '#FFEBEE'  # Soft Red (Pathogenic rule)
C_WARN_LINE   = '#C62828'  # Deep Red
C_DL_BG       = '#EDE7F6'  # Soft Purple (AI Engine)
C_DL_LINE     = '#512DA8'  # Purple
C_CAL_BG      = '#E0F2F1'  # Soft Teal (Platt)
C_CAL_LINE    = '#00796B'  # Teal
C_ZYG_BG      = '#FFF3E0'  # Soft Orange
C_ZYG_LINE    = '#E65100'  # Deep Orange
CLR_TEXT      = '#1A1A1A'
CLR_MUTED     = '#424242'

def draw_box(ax, x, y, w, h, bg, border, title, lines=None, tag=None, rx=0.06, title_color=CLR_TEXT):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                         facecolor=bg, edgecolor=border, linewidth=1.1, zorder=2)
    ax.add_patch(box)
    
    if lines:
        ax.text(x + w/2, y + h - 0.20, title, ha='center', va='center', fontsize=8.6, fontweight='bold', color=title_color, zorder=3)
        y_text = y + h - 0.40
        for l in lines:
            ax.text(x + w/2, y_text, l, ha='center', va='center', fontsize=7.1, color=CLR_MUTED, zorder=3)
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

def build_clinical_decision_diagram(out_dir):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Figure Header
    ax.text(8.0, 9.75, "FIGURE 3: Hybrid Clinical Decision Workflow, Platt Calibration & Penetrance Mapping",
            ha='center', va='center', fontsize=13.0, fontweight='bold', color='#111111')
    ax.text(8.0, 9.48, "Deterministic clinical safeguards, 5-seed neural ensemble inference, empirical calibration, and zygosity-aware risk assignment",
            ha='center', va='center', fontsize=8.6, style='italic', color='#555555')

    # STAGE 0: PATIENT VCF INGESTION
    draw_box(ax, 0.4, 7.8, 2.8, 1.4, C_INPUT_BG, C_INPUT_LINE, "Patient Genomic VCF",
             ["Input VCF (.vcf / .vcf.gz)",
              "Strict GRCh38 Reference Check",
              "Quality Filters: PASS",
              r"Read Depth $\mathrm{DP} \geq 10$, $\mathrm{GQ} \geq 20$"],
             tag="Sequencing Quality Control", rx=0.07)

    draw_arrow(ax, (3.2, 8.5), (3.8, 8.5), color=C_INPUT_LINE, lw=1.3)

    # -------------------------------------------------------------
    # 3 DETERMINISTIC CLINICAL STAGES
    # -------------------------------------------------------------
    # STAGE 1: POPULATION ALLELE FREQUENCY
    draw_box(ax, 3.8, 7.6, 3.4, 1.8, C_FILTER_BG, C_FILTER_LINE, "Stage I: Population Frequency",
             ["Query gnomAD Global Cohort",
              r"Condition: $\mathrm{AF} > 0.01$ (1.0%)?",
              "• YES: Common Neutral Polymorphism",
              "• NO: Advance to Clinical Curations"],
             tag=r"Deterministic Threshold", rx=0.07)

    # Branch: AF > 0.01 -> Benign
    draw_arrow(ax, (5.5, 7.6), (5.5, 6.4), color='#2E7D32', label="YES (AF > 1%)", label_pos=(0.6, 0))
    draw_box(ax, 4.2, 5.2, 2.6, 1.2, C_PASS_BG, C_PASS_LINE, "Verdict: Benign",
             ["Common Polymorphism", r"Calibrated Score: $P = 0.001$", "Zero Clinical Penetrance"],
             tag="Resolved: Benign", rx=0.06, title_color='#1B5E20')

    # Next Stage
    draw_arrow(ax, (7.2, 8.5), (7.8, 8.5), color='#C62828', label="NO", label_pos=(0, 0.15))

    # STAGE 2: CLINVAR CONSENSUS LOOKUP
    draw_box(ax, 7.8, 7.6, 3.5, 1.8, C_FILTER_BG, C_FILTER_LINE, "Stage II: ClinVar Expert Consensus",
             ["Index: (chr, pos, ref, alt)",
              r"Evidence Stars $\geq 2$ Stars?",
              "(Multiple submitters / Expert panel)",
              "• YES: Adopt Curated Consensus",
              "• NO: Novel or 1-Star/Conflicting"],
             tag="Database Ground-Truth", rx=0.07)

    # Branch: Stars >= 2 -> ClinVar Verdict
    draw_arrow(ax, (9.55, 7.6), (9.55, 6.4), color='#C62828', label="YES (Stars >= 2)", label_pos=(0.7, 0))
    draw_box(ax, 8.1, 5.2, 2.9, 1.2, C_WARN_BG, C_WARN_LINE, "Adopt ClinVar Verdict",
             [r"Pathogenic: $P = 0.999$", r"Benign: $P = 0.001$", r"VUS: $P = 0.500$"],
             tag="Curated Consensus", rx=0.06, title_color='#B71C1C')

    # Next Stage
    draw_arrow(ax, (11.3, 8.5), (11.9, 8.5), color='#C62828', label="NO", label_pos=(0, 0.15))

    # STAGE 3: RULE-BASED LOF & SPLICING SAFEGUARDS
    draw_box(ax, 11.9, 7.4, 3.7, 2.0, C_FILTER_BG, C_FILTER_LINE, "Stage III: Structural LoF & Splicing",
             ["Structural Rupture Rules:",
              "1. Frameshift or Canonical Splice (±1,2 bp)",
              r"2. SpliceAI $\mathrm{DS\_MAX} \geq 0.50$ (Cryptic)",
              "• YES: Structural Inactivation Rule",
              "• NO: Unclassified SNV"],
             tag="Rule-Based Inactivation", rx=0.07)

    # Branch: LoF Rule -> Likely Pathogenic
    draw_arrow(ax, (13.75, 7.4), (13.75, 6.4), color='#C62828', label="YES (LoF)", label_pos=(0.6, 0))
    draw_box(ax, 12.3, 5.2, 2.9, 1.2, C_WARN_BG, C_WARN_LINE, "Verdict: Likely Pathogenic",
             ["Predicted Loss-of-Function", r"Calibrated Score: $P = 0.950$", "Bypasses Protein Model"],
             tag="Resolved: LoF Rule", rx=0.06, title_color='#B71C1C')

    # -------------------------------------------------------------
    # STAGE 4: DEEP NEURAL NETWORK (FUSIONRESNET ENSEMBLE)
    # -------------------------------------------------------------
    draw_arrow(ax, (13.75, 5.2), (13.75, 4.4), color='#455A64')
    draw_arrow(ax, (11.9, 8.2), (0.8, 3.8), color='#512DA8', curved=-0.4, label="Unclassified SNVs", label_pos=(-1.0, 1.8))

    draw_box(ax, 0.4, 2.3, 4.6, 1.9, C_DL_BG, C_DL_LINE, "Stage IV: FusionResNet Ensemble (5 Seeds)",
             ["Multimodal Feature Fusion (Tabular + ESM-2)",
              "Multi-Seed Averaging (Seeds 42, 101, 2024, 7, 999):",
              r"  $z_{\mathrm{ensemble}} = \frac{1}{K} \sum_{k=1}^{K} z^{(k)}$",
              "Out-of-Gene AUC = 0.9919 [95% CI: 0.9913–0.9925]"],
             tag=r"Raw Logit: $z_{\mathrm{ensemble}} \in \mathbb{R}$", rx=0.07)

    # -------------------------------------------------------------
    # STAGE 5: PLATT TEMPERATURE SCALING
    # -------------------------------------------------------------
    draw_arrow(ax, (5.0, 3.25), (5.7, 3.25), color=C_DL_LINE, lw=1.3)

    draw_box(ax, 5.7, 2.3, 4.5, 1.9, C_CAL_BG, C_CAL_LINE, "Stage V: Empirical Platt Calibration",
             [r"Optimized over Validation Logits via L-BFGS:",
              r"  $z_{\mathrm{cal}} = s \cdot z_{\mathrm{ensemble}} + b$",
              r"  ($s = 1.0559$, equivalent $T = 0.947$, $b = -1.5561$)",
              r"Posterior Probability: $P = \sigma(z_{\mathrm{cal}})$",
              r"Expected Calibration Error: $\mathrm{ECE}: 0.0279 \to 0.0009$"],
             tag=r"Calibrated Probability: $P \in [0, 1]$", rx=0.07)

    # -------------------------------------------------------------
    # STAGE 6: CLINICAL TIER ASSIGNMENT
    # -------------------------------------------------------------
    draw_arrow(ax, (10.2, 3.25), (10.9, 3.25), color=C_CAL_LINE, lw=1.3)

    draw_box(ax, 10.9, 2.1, 4.7, 2.3, C_FILTER_BG, C_FILTER_LINE, "Stage VI: ACMG/AMP Tier Assignment",
             [r"• $P \geq 0.90 \to$ Pathogenic",
              r"• $0.70 \leq P < 0.90 \to$ Likely Pathogenic",
              r"• $0.30 \leq P < 0.70 \to$ Variant of Uncertain Significance (VUS)",
              r"• $0.10 \leq P < 0.30 \to$ Likely Benign",
              r"• $P < 0.10 \to$ Benign"],
             tag="Categorical Decision Boundaries", rx=0.07)

    # -------------------------------------------------------------
    # STAGE 7: ZYGOSITY & PENETRANCE ENGINE (BOTTOM)
    # -------------------------------------------------------------
    draw_arrow(ax, (13.25, 2.1), (13.25, 1.5), color=C_ZYG_LINE, lw=1.3)

    draw_box(ax, 0.4, 0.15, 15.2, 1.35, C_ZYG_BG, C_ZYG_LINE, "Stage VII: Zygosity Resolution & Curated Lifetime Penetrance Mapping (Hospital PDF Output)",
             [r"1. Autosomal Dominant Genes (e.g., BRCA1, BRCA2, TP53, MLH1, MSH2): Monoallelic Pathogenic $\to$ Immediate High-Penetrance Assignment",
              r"2. Autosomal Recessive Genes (e.g., MUTYH, NTHL1): Monoallelic Heterozygous $\to$ Reclassified as 'Carrier (Recessive)', Biallelic/Compound Het Required for High Risk",
              r"3. Curated Penetrance Lookup (data/penetrance.csv): JAMA/Lancet/NEJM peer-reviewed lifetime cancer risk ranges vs. general population baseline",
              r"4. Output Deliverables: Structured JSON record, Plaintext clinical summary, Publication-grade Hospital PDF Report with Visual Risk Radar"],
             tag="Final Clinical Translation: Actionable Risk Stratification", rx=0.05, title_color='#BF360C')

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "figure3_clinical_decision_workflow.pdf")
    svg_path = os.path.join(out_dir, "figure3_clinical_decision_workflow.svg")
    png_path = os.path.join(out_dir, "figure3_clinical_decision_workflow.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    fig.savefig(svg_path, format="svg", bbox_inches='tight')
    fig.savefig(png_path, format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 3 (Clinical Decision Workflow) generated:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\monoj\OneDrive\Desktop\deepL\Deep-Learning-project\research-figures"
    build_clinical_decision_diagram(out_dir)
