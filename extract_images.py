import zipfile
from PIL import Image
import numpy as np
import os

# Path to your ZIP file
zip_path = '/zhome/31/c/147318/Advaned_DLCV/exam_project/archive.zip'

# Destination directory for extracted images (temporary)
destination_dir = './data_frontal/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


# Base path within the ZIP to start searching for images
base_path = 'CheXpert-v1.0-small/train/'

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Initialize a counter for the number of files processed
    count = 0
    
    # Iterate through each file in the ZIP archive
    for f in zip_ref.namelist():
        # Check if the file matches the criteria and increment the counter if it does
        if f.startswith(base_path) and f.endswith('_frontal.jpg') and not f.startswith('.') and not f.endswith('._view1_frontal.jpg') and count < 10000:
            # Extract the specific file to the destination directory
            extracted_file_path = zip_ref.extract(f, destination_dir)
            
            # Convert extracted file path to a format compatible with PIL
            with Image.open(extracted_file_path) as img:
                # Convert the image to a NumPy array
                img_array = np.array(img)
                
                # Define the output path for the .npy file, replacing directory separators with underscores
                npy_file_name = f.replace('/', '_').replace('.jpg', '.npy')
                npy_file_path = os.path.join(destination_dir, npy_file_name)

                
                # Save the NumPy array to a .npy file
                np.save(npy_file_path, img_array)
            
            # Delete the extracted JPEG file to save space
            os.remove(extracted_file_path)
            
            # Increment the counter
            count += 1
            
            # Break the loop if 10,000 files have been processed
            if count >= 10000:
                break

print(f"Processing of up to {count} images completed.")
