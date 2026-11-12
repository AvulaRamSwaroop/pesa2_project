"""
comparisons/generate_presentation_pdfs.py
Stitches the 'Obtained' and 'True' plot PNGs side-by-side into formatted presentation PDFs.
"""
import os
import glob
from PIL import Image

def create_side_by_side_pdf(plot_dir, output_pdf_name):
    if not os.path.exists(plot_dir):
        print(f"[!] Directory not found: {plot_dir}")
        return

    print(f"\n--- Scanning {plot_dir} for plot pairs ---")
    
    # Find all 'obtained' images (works for both pareto_front and pareto_set)
    obtained_files = sorted(glob.glob(os.path.join(plot_dir, "*_obtained.png")))
    
    if not obtained_files:
        print("[!] No '*_obtained.png' files found in this directory.")
        return

    pdf_pages = []
    
    for obt_path in obtained_files:
        # Match it with the corresponding 'true' plot
        true_path = obt_path.replace("_obtained.png", "_true.png")
        
        if os.path.exists(true_path):
            try:
                # Load images
                img_obt = Image.open(obt_path).convert("RGB")
                img_true = Image.open(true_path).convert("RGB")
                
                # Calculate dimensions for side-by-side layout
                total_width = img_obt.width + img_true.width
                max_height = max(img_obt.height, img_true.height)
                
                # Create a blank white canvas
                combined_page = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                
                # Paste them side-by-side (Obtained on left, True on right to match your reference)
                combined_page.paste(img_obt, (0, 0))
                combined_page.paste(img_true, (img_obt.width, 0))
                
                pdf_pages.append(combined_page)
                
                benchmark_name = os.path.basename(obt_path).replace('_obtained.png', '')
                print(f"Stitched {benchmark_name}...")
                
            except Exception as e:
                print(f"[!] Error processing {obt_path}: {e}")

    # Save all stitched images into a single multi-page PDF
    if pdf_pages:
        output_path = os.path.join(plot_dir, output_pdf_name)
        # The save_all=True and append_images commands bundle everything into a single file
        pdf_pages[0].save(output_path, save_all=True, append_images=pdf_pages[1:])
        print(f"\n[+] Success! Formatted PDF saved to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Generate the PDF for the 30 Standard Benchmarks
    standard_plots_dir = os.path.join(base_dir, "30_benchmarks", "plots")
    create_side_by_side_pdf(standard_plots_dir, "StandardSuite_SideBySide_Comparison.pdf")
    
    # 2. Generate the PDF for the CEC2020 MMO Benchmarks
    mmo_plots_dir = os.path.join(base_dir, "mmo2020_benchmarks", "plots")
    create_side_by_side_pdf(mmo_plots_dir, "MMO2020_SideBySide_Comparison.pdf")