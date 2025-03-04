import os
import sys
from PIL import Image
import pillow_heif
from pathlib import Path

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()
input_folder = r"C:\Users\MichaelDoman\OneDrive - eMPiGO Technologies\Pictures\PrimorisSitePhotos"
output_folder = r"C:\Users\MichaelDoman\OneDrive - eMPiGO Technologies\Pictures\PrimorisSitePhotos\JPG"
def convert_heic_to_jpg(input_folder, output_folder):
    """
    Convert all HEIC images in input_folder to JPG format
    
    Args:
        input_folder (str): Path to folder containing HEIC images
        output_folder (str, optional): Path to save converted JPG images. 
                                      If None, saves in the same folder
    """
    # Create output folder if specified and doesn't exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = input_folder
        
    # Get all HEIC files in the input folder
    heic_files = list(Path(input_folder).glob("*.HEIC"))
    heic_files.extend(list(Path(input_folder).glob("*.heic")))
    
    if not heic_files:
        print("No HEIC files found in the specified folder.")
        return
    
    # Convert each file
    for heic_file in heic_files:
        try:
            # Open the HEIC file
            img = Image.open(heic_file)
            
            # Create output file path
            jpg_filename = os.path.splitext(os.path.basename(heic_file))[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)
            
            # Convert and save as JPG
            img.convert("RGB").save(jpg_path, "JPEG")
            print(f"Converted: {heic_file} -> {jpg_path}")
            
        except Exception as e:
            print(f"Error converting {heic_file}: {e}")

if __name__ == "__main__":
    # If command line arguments provided
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
        convert_heic_to_jpg(input_folder, output_folder)
    else:
        # Interactive mode
        input_folder = input("Enter the folder path containing HEIC images: ")
        output_choice = input("Save JPG images in the same folder? (y/n): ").lower()
        
        if output_choice == "n":
            output_folder = input("Enter output folder path for JPG images: ")
        else:
            output_folder = None
            
        convert_heic_to_jpg(input_folder, output_folder)