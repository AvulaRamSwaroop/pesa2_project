"""
visualization/plots.py
Generates individual PNG plots and builds the side-by-side figure for the master PDF.
"""
import os
import matplotlib.pyplot as plt

def create_and_save_plots(name, obtained, true_pf, is_3d, timestamp):
    """Generates 3 separate PNGs and returns a side-by-side figure for the PDF."""
    
    # Create the nested directory structure for individual PNGs
    save_dir = os.path.join("plots", name, "pngs")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot & Save: Obtained Front
    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(111, projection='3d' if is_3d else None)
    if is_3d:
        ax1.scatter(obtained[:, 0], obtained[:, 1], obtained[:, 2], s=8, color="#0055b3")
        ax1.set_zlabel("$f_3$")
    else:
        ax1.scatter(obtained[:, 0], obtained[:, 1], s=8, color="#0055b3")
    ax1.set_title(f"{name} - Obtained Front")
    ax1.set_xlabel("$f_1$")
    ax1.set_ylabel("$f_2$")
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, f"{timestamp}_obtained.png"), dpi=200)
    plt.close(fig1)

    # 2. Plot & Save: True Front
    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111, projection='3d' if is_3d else None)
    if is_3d:
        ax2.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=8, color="#d92027")
        ax2.set_zlabel("$f_3$")
    else:
        ax2.scatter(true_pf[:, 0], true_pf[:, 1], s=8, color="#d92027")
    ax2.set_title(f"{name} - True Pareto Front")
    ax2.set_xlabel("$f_1$")
    ax2.set_ylabel("$f_2$")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"{timestamp}_true.png"), dpi=200)
    plt.close(fig2)

    # 3. Plot & Save: Overlay Comparison
    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection='3d' if is_3d else None)
    if is_3d:
        ax3.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=8, color="#d92027", label="True", alpha=0.3)
        ax3.scatter(obtained[:, 0], obtained[:, 1], obtained[:, 2], s=8, color="#0055b3", label="Obtained")
        ax3.set_zlabel("$f_3$")
    else:
        ax3.scatter(true_pf[:, 0], true_pf[:, 1], s=8, color="#d92027", label="True", alpha=0.3)
        ax3.scatter(obtained[:, 0], obtained[:, 1], s=8, color="#0055b3", label="Obtained")
    ax3.set_title(f"{name} - Comparison")
    ax3.set_xlabel("$f_1$")
    ax3.set_ylabel("$f_2$")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, f"{timestamp}_comparison.png"), dpi=200)
    plt.close(fig3)

    # 4. Create Side-by-Side Figure for the Master PDF (Matches your reference PDF)
    fig_pdf = plt.figure(figsize=(12, 5))
    
    # Left: Obtained
    ax_pdf1 = fig_pdf.add_subplot(1, 2, 1, projection='3d' if is_3d else None)
    if is_3d:
        ax_pdf1.scatter(obtained[:, 0], obtained[:, 1], obtained[:, 2], s=6, color="#0055b3", label="Final Pareto front")
        ax_pdf1.set_zlabel("$f_3$")
    else:
        ax_pdf1.scatter(obtained[:, 0], obtained[:, 1], s=8, color="#0055b3", label="Final Pareto front")
    ax_pdf1.set_title(f"{name} - Bit-Level Neighborhood")
    ax_pdf1.set_xlabel("$f_1$")
    ax_pdf1.set_ylabel("$f_2$")
    ax_pdf1.grid(True, linestyle="--", alpha=0.5)
    ax_pdf1.legend(loc="upper right")

    # Right: True
    ax_pdf2 = fig_pdf.add_subplot(1, 2, 2, projection='3d' if is_3d else None)
    if is_3d:
        ax_pdf2.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=6, color="#d92027", label="Reference Pareto Front")
        ax_pdf2.set_zlabel("$f_3$")
    else:
        ax_pdf2.scatter(true_pf[:, 0], true_pf[:, 1], s=8, color="#d92027", label="Reference Pareto Front")
    ax_pdf2.set_title(f"{name} - Reference Pareto Front")
    ax_pdf2.set_xlabel("$f_1$")
    ax_pdf2.set_ylabel("$f_2$")
    ax_pdf2.grid(True, linestyle="--", alpha=0.5)
    ax_pdf2.legend(loc="upper right")

    fig_pdf.tight_layout()
    
    return fig_pdf