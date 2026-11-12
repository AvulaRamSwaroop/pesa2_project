"""
comparisons/generate_presentation_pdfs.py
Stitches individual True and Obtained PNG plots side-by-side into a Master PDF,
replicating the exact landscape format of the reference document.
"""
import os
import sys
try:
    from PIL import Image
except ImportError:
    print("Pillow library is missing. Please run: pip install Pillow")
    sys.exit(1)

def stitch_side_by_side(left_path, right_path):
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        return None
    
    img_left = Image.open(left_path)
    img_right = Image.open(right_path)
    
    w1, h1 = img_left.size
    w2, h2 = img_right.size
    
    # Add a white margin to mimic a proper PDF document page (like the reference)
    margin = 50
    spacing = 20
    total_width = w1 + w2 + spacing + (2 * margin)
    total_height = max(h1, h2) + (2 * margin)
    
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    canvas.paste(img_left, (margin, margin))
    canvas.paste(img_right, (margin + w1 + spacing, margin))
    
    return canvas

def build_master_pdf(directory, output_filename, plot_types=["pareto_front"]):
    if not os.path.exists(directory):
        print(f"Directory {directory} not found. Skipping.")
        return

    pages = []
    
    # Dynamically gather all problem names in the folder
    all_files = os.listdir(directory)
    prob_names = set()
    for f in all_files:
        if f.endswith("_obtained.png"):
            # Extract the benchmark name (e.g., "ZDT1" from "ZDT1_pareto_front_obtained.png")
            name = f.split("_pareto_")[0]
            prob_names.add(name)
            
    if not prob_names:
        print(f"No PNGs found in {directory}.")
        return

    # Sort alphabetically to keep ZDT, DTLZ, etc. grouped together
    for prob in sorted(prob_names):
        for ptype in plot_types:
            obt_file = f"{prob}_{ptype}_obtained.png"
            true_file = f"{prob}_{ptype}_true.png"
            
            obt_path = os.path.join(directory, obt_file)
            true_path = os.path.join(directory, true_file)
            
            canvas = stitch_side_by_side(obt_path, true_path)
            if canvas:
                pages.append(canvas)
            else:
                print(f"Missing images to stitch for {prob} ({ptype})")
            
    if pages:
        out_path = os.path.join(directory, output_filename)
        # Save all canvases into a single multi-page PDF
        pages[0].save(out_path, save_all=True, append_images=pages[1:])
        print(f"[+] Successfully generated {output_filename} ({len(pages)} pages) in {directory}")

if __name__ == "__main__":
    print("Stitching images into exact replica side-by-side PDFs...")
    
    # 1. Process the 30 Standard Benchmarks (Objective Space only)
    build_master_pdf(
        os.path.join("comparisons", "30_benchmarks", "plots"), 
        "Standard_30_Replica.pdf", 
        plot_types=["pareto_front"]
    )
    
    # 2. Process the 24 MMO 2020 Benchmarks (Objective Space AND Decision Space)
    build_master_pdf(
        os.path.join("comparisons", "mmo2020_benchmarks", "plots"), 
        "MMO2020_Replica.pdf", 
        plot_types=["pareto_front", "pareto_set"]
    )
