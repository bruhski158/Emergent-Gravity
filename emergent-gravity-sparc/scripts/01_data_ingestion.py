import os
import glob
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

sys.path.append(project_root)

from src.io import load_and_convert_sparc, save_processed_data

def main():
    print("--- Starting Step 1: SPARC Data Ingestion & Conversion ---")
    
    raw_dir = os.path.join(project_root, "data", "raw", "Rotmod_LTG")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.dat"))
    
    if not raw_files:
        print(f"Error: No .dat files found in '{raw_dir}'.")
        print("Please extract the Rotmod_LTG.zip contents into that folder.")
        return

    print(f"Found {len(raw_files)} galaxy files. Processing...")
    
    success_count = 0
    
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        galaxy_name = filename.split('_')[0] 
        
        try:
            df_clean = load_and_convert_sparc(filepath)
            
            save_processed_data(df_clean, galaxy_name, processed_dir)
            success_count += 1
            
        except Exception as e:
            print(f"Failed to process {galaxy_name}: {e}")
            
    print(f"\n✅ Success! Processed {success_count}/{len(raw_files)} galaxies.")
    print(f"Cleaned SI-unit CSVs are now stored in '{processed_dir}'.")

if __name__ == "__main__":
    main()