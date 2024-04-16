import zipfile
from PIL import Image
import numpy as np
import os
import csv

# Path to your ZIP file
zip_path = '/zhome/31/c/147318/Advaned_DLCV/exam_project/ADLCV_AnomalyDetection/archive.zip'

# Path to the CSV file
csv_path = '/zhome/31/c/147318/Advaned_DLCV/exam_project/ADLCV_AnomalyDetection/che_pm_shortcut_labels.csv'

# Base path within the ZIP to start searching for images
base_path = 'CheXpert-v1.0-small/train/'

# Destination directory for extracted images
destination_dir = './data_frontal/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Read the CSV to get classification info
image_labels = {}
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['PM'] == '0':  # Only consider images without pacemakers
            image_labels[row['img_name']] = row['Cardiomegaly']

# Counters for images with and without cardiomegaly
cardio_count = 0
non_cardio_count = 0

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Iterate through each file in the ZIP archive
    for f in zip_ref.namelist():
        if f.startswith(base_path) and f.endswith('_frontal.jpg') and f.split('/')[-1] in image_labels:
            cardiomegaly = image_labels[f.split('/')[-1]]
            
            # Check and update counts accordingly
            if cardiomegaly == '1' and cardio_count < 5000:
                cardio_count += 1
            elif cardiomegaly == '0' and non_cardio_count < 5000:
                non_cardio_count += 1
            else:
                continue  # Skip if either count exceeds 5000
            
            # Extract and process image
            subfolder = 'Cardiomegaly_' + cardiomegaly
            final_path = os.path.join(destination_dir, subfolder)
            if not os.path.exists(final_path):
                os.makedirs(final_path)
            extracted_file_path = zip_ref.extract(f, final_path)
            
            with Image.open(extracted_file_path) as img:
                img_array = np.array(img)
                npy_file_name = f.replace('/', '_').replace('.jpg', '.npy')
                npy_file_path = os.path.join(final_path, npy_file_name)
                np.save(npy_file_path, img_array)
            
            os.remove(extracted_file_path)
            
            # Stop processing if both counts reach 5000
            if cardio_count >= 5000 and non_cardio_count >= 5000:
                break

print("Processing completed.")