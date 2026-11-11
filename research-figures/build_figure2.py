"""
GenomeDL v3 - Publication Figure 2 Generator
IEEE Style 3D Isometric Layer-Wise Tensor Representation
Illustrates exact tensor transformations:
 - Tabular: [B, 26] -> [B, 64]
 - Categoricals: [B, 1] + [B, 1] -> [B, 8] + [B, 8] -> [B, 16]
 - ESM Branch: [B, 480] -> [B, 64] + [B, 2] -> [B, 64]
 - Multimodal Concatenation: [B, 144]
 - Dense Projection: [B, 256]
 - ResNet Residual Blocks (x + f(x)): [B, 256] -> [B, 256]
 - Classification Logit: [B, 1] -> Calibrated Probability in [0, 1]
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Publication Typography
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'cm'

def draw_prism(ax, x, y, z, dx, dy, dz, face_color, edge_color='#222222', alpha=0.85, lw=0.9):
    """Draws a 3D rectangular cuboid representing a tensor representation."""
    # 8 vertices
    v = [
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ]
    # 6 faces
    faces = [
        [v[0], v[1], v[2], v[3]], # bottom
        [v[4], v[5], v[6], v[7]], # top
        [v[0], v[1], v[5], v[4]], # front
        [v[2], v[3], v[7], v[6]], # back
        [v[1], v[2], v[6], v[5]], # right
        [v[0], v[3], v[7], v[4]]  # left
    ]
    poly = Poly3DCollection(faces, facecolors=face_color, edgecolors=edge_color,
                            linewidths=lw, alpha=alpha, zorder=5)
    ax.add_collection3d(poly)

def draw_3d_arrow(ax, p1, p2, color='#444444', lw=1.2):
    """Draws an informative 3D vector connector line with end marker."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, linewidth=lw, linestyle='-', zorder=10)
    ax.scatter([p2[0]], [p2[1]], [p2[2]], color=color, s=20, marker='o', zorder=11)

def build_figure_2(out_dir):
    fig = plt.figure(figsize=(15.0, 10.0), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=22, azim=-55)

    # Clean axes styling for academic IEEE presentation
    ax.set_facecolor('white')
    ax.grid(False)
    # Hide axis lines and ticks
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # IEEE Color definitions
    C_TAB_IN   = '#B3E5FC'  # Light blue
    C_TAB_OUT  = '#0288D1'  # Deep sky blue
    C_CAT_IN   = '#FFF9C4'  # Pale yellow
    C_CAT_OUT  = '#FBC02D'  # Deep amber
    C_ESM_IN   = '#D1C4E9'  # Lavender
    C_ESM_OUT  = '#512DA8'  # Deep purple
    C_FUSION   = '#FFE0B2'  # Orange
    C_FUS_OUT  = '#E65100'  # Deep Orange
    C_RES1     = '#E1BEE7'  # Soft magenta
    C_RES2     = '#BA68C8'  # Medium magenta
    C_HEAD     = '#C8E6C9'  # Soft Green
    C_PRED     = '#2E7D32'  # Dark Green

    # -------------------------------------------------------------
    # LAYER 1: MULTIMODAL INPUT TENSORS (X = 0)
    # -------------------------------------------------------------
    # Tabular Input: [B, 26]
    draw_prism(ax, x=0, y=14, z=5, dx=0.8, dy=2.6, dz=1.2, face_color=C_TAB_IN, edge_color='#0277BD')
    ax.text(0, 15.3, 6.7, "Tabular\n[B, 26]", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#01579B')

    # Categorical Inputs: [B, 1] + [B, 1]
    draw_prism(ax, x=0, y=9.5, z=5.5, dx=0.8, dy=0.8, dz=0.8, face_color=C_CAT_IN, edge_color='#F57F17')
    ax.text(0, 9.9, 6.7, "Variant Class\n[B, 1]", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#E65100')

    draw_prism(ax, x=0, y=7.8, z=5.5, dx=0.8, dy=0.8, dz=0.8, face_color=C_CAT_IN, edge_color='#F57F17')
    ax.text(0, 8.2, 6.7, "Syndrome\n[B, 1]", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#E65100')

    # ESM Inputs: [B, 1] LLR + [B, 1] Avail + [B, 480] Embedding
    draw_prism(ax, x=0, y=2.0, z=4.0, dx=0.8, dy=4.0, dz=3.2, face_color=C_ESM_IN, edge_color='#4A148C')
    ax.text(0, 4.0, 7.8, "ESM-2 Tensor\nLLR(1)+Avail(1)+Emb(480)", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#311B92')

    # -------------------------------------------------------------
    # LAYER 2: BRANCH PROJECTIONS & SUB-NETWORKS (X = 3.5)
    # -------------------------------------------------------------
    # Tabular Linear+LN+GELU -> [B, 64]
    draw_prism(ax, x=3.5, y=13.5, z=4.8, dx=1.0, dy=3.5, dz=1.6, face_color=C_TAB_OUT, edge_color='#01579B')
    ax.text(4.0, 15.2, 6.9, "h_tab: Linear(26→64)\n+ LN + GELU [B, 64]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#01579B')
    draw_3d_arrow(ax, (0.8, 15.3, 5.6), (3.5, 15.2, 5.6), color='#0277BD')

    # Categorical Embedding Lookup -> [B, 8] + [B, 8] -> [B, 16]
    draw_prism(ax, x=3.5, y=8.2, z=5.1, dx=1.0, dy=1.8, dz=1.0, face_color=C_CAT_OUT, edge_color='#E65100')
    ax.text(4.0, 9.1, 6.6, "h_cat: Embeddings\n[B, 8]+[B, 8] = [B, 16]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#BF360C')
    draw_3d_arrow(ax, (0.8, 9.9, 5.9), (3.5, 9.1, 5.6), color='#F57F17')
    draw_3d_arrow(ax, (0.8, 8.2, 5.9), (3.5, 9.1, 5.6), color='#F57F17')

    # ESM MLP Subnetwork -> [B, 64]
    draw_prism(ax, x=3.5, y=2.5, z=4.8, dx=1.0, dy=3.5, dz=1.6, face_color=C_ESM_OUT, edge_color='#311B92')
    ax.text(4.0, 4.25, 6.9, "h_esm: Linear(480→64)\n+ MLP(66→64) [B, 64]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#311B92')
    draw_3d_arrow(ax, (0.8, 4.0, 5.6), (3.5, 4.25, 5.6), color='#4A148C')

    # -------------------------------------------------------------
    # LAYER 3: MULTIMODAL CONCATENATION & PROJECTION (X = 7.5)
    # -------------------------------------------------------------
    # Concat: 64 + 16 + 64 = 144
    draw_prism(ax, x=7.2, y=6.0, z=4.4, dx=1.0, dy=6.5, dz=2.4, face_color=C_FUSION, edge_color='#E65100')
    ax.text(7.7, 9.25, 7.3, "Concatenation Tensor\n[B, 144] (64+16+64)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#E65100')

    draw_3d_arrow(ax, (4.5, 15.2, 5.6), (7.2, 10.5, 5.6), color='#E65100')
    draw_3d_arrow(ax, (4.5, 9.1, 5.6), (7.2, 9.25, 5.6), color='#E65100')
    draw_3d_arrow(ax, (4.5, 4.25, 5.6), (7.2, 8.0, 5.6), color='#E65100')

    # Fusion Linear(144 -> 256) + LN + GELU
    draw_prism(ax, x=9.4, y=5.5, z=4.0, dx=1.2, dy=7.5, dz=3.2, face_color=C_FUS_OUT, edge_color='#BF360C')
    ax.text(10.0, 9.25, 7.8, "Fusion Dense In\nLinear(144→256) [B, 256]", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#BF360C')
    draw_3d_arrow(ax, (8.2, 9.25, 5.6), (9.4, 9.25, 5.6), color='#E65100', lw=1.6)

    # -------------------------------------------------------------
    # LAYER 4: RESIDUAL BLOCKS 1 & 2 (X = 12.0 & X = 15.0)
    # -------------------------------------------------------------
    # ResBlock 1: [B, 256]
    draw_prism(ax, x=12.0, y=5.5, z=4.0, dx=1.4, dy=7.5, dz=3.2, face_color=C_RES1, edge_color='#6A1B9A')
    ax.text(12.7, 9.25, 7.8, "Residual Block 1\n(2x LN+GELU+Drop+Dense)\n[B, 256]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#4A148C')
    draw_3d_arrow(ax, (10.6, 9.25, 5.6), (12.0, 9.25, 5.6), color='#6A1B9A', lw=1.6)

    # Skip 1 loop
    ax.plot([10.6, 11.3, 13.4, 13.4], [9.25, 14.5, 14.5, 9.25], [7.2, 7.5, 7.5, 7.2],
            color='#8E24AA', linestyle='--', linewidth=1.3, zorder=12)
    ax.text(12.3, 14.7, 7.7, "+ Skip (x)", ha='center', va='bottom', fontsize=7.2, color='#8E24AA', fontweight='bold')

    # ResBlock 2: [B, 256]
    draw_prism(ax, x=15.0, y=5.5, z=4.0, dx=1.4, dy=7.5, dz=3.2, face_color=C_RES2, edge_color='#4A148C')
    ax.text(15.7, 9.25, 7.8, "Residual Block 2\n(2x LN+GELU+Drop+Dense)\n[B, 256]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#311B92')
    draw_3d_arrow(ax, (13.4, 9.25, 5.6), (15.0, 9.25, 5.6), color='#4A148C', lw=1.6)

    # Skip 2 loop
    ax.plot([13.4, 14.2, 16.4, 16.4], [9.25, 14.5, 14.5, 9.25], [7.2, 7.5, 7.5, 7.2],
            color='#6A1B9A', linestyle='--', linewidth=1.3, zorder=12)
    ax.text(15.3, 14.7, 7.7, "+ Skip (x)", ha='center', va='bottom', fontsize=7.2, color='#6A1B9A', fontweight='bold')

    # -------------------------------------------------------------
    # LAYER 5: PREDICTION HEAD & CALIBRATION (X = 18.0)
    # -------------------------------------------------------------
    # Dense Head: [B, 1]
    draw_prism(ax, x=18.0, y=8.5, z=5.0, dx=0.8, dy=1.5, dz=1.2, face_color=C_HEAD, edge_color='#2E7D32')
    ax.text(18.4, 9.25, 6.7, "Head Logit\nz: [B, 1]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#1B5E20')
    draw_3d_arrow(ax, (16.4, 9.25, 5.6), (18.0, 9.25, 5.6), color='#2E7D32', lw=1.5)

    # Platt Scaling + Sigmoid Output: [B, 1]
    draw_prism(ax, x=20.2, y=8.7, z=5.2, dx=0.8, dy=1.1, dz=0.8, face_color=C_PRED, edge_color='#1B5E20')
    ax.text(20.6, 9.25, 6.5, "P(Pathogenic)\nin [0.0, 1.0]", ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#1B5E20')
    draw_3d_arrow(ax, (18.8, 9.25, 5.6), (20.2, 9.25, 5.6), color='#1B5E20', lw=1.5)

    # Title & Caption Overlay
    ax.text2D(0.50, 0.96, "FIGURE 2: 3D Isometric Layer-Wise Tensor Transformation Pipeline",
              transform=ax.transAxes, ha='center', va='center', fontsize=13.0, fontweight='bold', color='#111111')
    ax.text2D(0.50, 0.925, "Layer-by-layer spatial mapping illustrating dimensional projections from heterogeneous inputs [26, 16, 482] to the calibrated risk score",
              transform=ax.transAxes, ha='center', va='center', fontsize=8.8, style='italic', color='#444444')

    # Limits and framing
    ax.set_xlim(-1, 22)
    ax.set_ylim(0, 18)
    ax.set_zlim(2, 9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "figure2_layerwise_3d_architecture.pdf")
    svg_path = os.path.join(out_dir, "figure2_layerwise_3d_architecture.svg")
    png_path = os.path.join(out_dir, "figure2_layerwise_3d_architecture.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
    fig.savefig(svg_path, format="svg", bbox_inches='tight')
    fig.savefig(png_path, format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated Figure 2 successfully:\n - {pdf_path}\n - {svg_path}\n - {png_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\monoj\OneDrive\Desktop\deepL\Deep-Learning-project\research-figures"
    build_figure_2(out_dir)
