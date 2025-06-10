import zipfile
import os
from PIL import Image
import io

# Path to your Excel file
excel_path = r"C:\Users\sukri\Downloads\porsche.xlsx"  # <-- update this with your actual file name
output_folder = "extracted_logos_porsche"
os.makedirs(output_folder, exist_ok=True)

# Open the Excel file as a ZIP archive
with zipfile.ZipFile(excel_path, 'r') as zip_ref:
    image_files = [file for file in zip_ref.namelist() 
                   if file.startswith('xl/media/') and not file.endswith('/')]
    
    for index, file in enumerate(image_files):
        filename = os.path.basename(file)
        if filename:  # Ensure it's a file, not a folder
            # Read the image data
            image_data = zip_ref.read(file)
            try:
                # Open the image data with PIL
                image = Image.open(io.BytesIO(image_data))


                custom_filename = f"porsche1_{index + 1}.png"

                target_path = os.path.join(output_folder, custom_filename)
                image.save(target_path, 'PNG')
                print(f"Converted {filename} to {custom_filename}")
                
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")

print(f"✅ Extracted all embedded images to: {output_folder} with panasonic names")
