import os
import pandas as pd
import numpy as np
from nilearn import image, masking
from nilearn.masking import compute_epi_mask

# Define your input and output directories
input_dir = "/Users/shayelshams/Desktop/Paper2025/DataRaw"         # Folder containing subject folders (e.g., sub-01, sub-02, etc.)
output_dir = "/Users/shayelshams/Desktop/Paper2025/dataextracted"            # Folder to save the CSV files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all subject folders (assuming they start with "sub-")
subject_ids = [folder for folder in os.listdir(input_dir) if folder.startswith("sub-")]

for subject in subject_ids:
    subject_folder = os.path.join(input_dir, subject)
    func_dir = os.path.join(subject_folder, "func")
    anat_dir = os.path.join(subject_folder, "anat")
    
    # Construct the file path for the fMRI data; adjust the naming pattern if needed.
    fmri_file = os.path.join(func_dir, f"{subject}_task-rest_bold.nii.gz")
    
    if not os.path.exists(fmri_file):
        print(f"Missing fMRI file for {subject}")
        continue

    # Load the fMRI image
    fmri_img = image.load_img(fmri_file)
    
    # Check if a brain mask exists; if not, compute one.
    mask_file = os.path.join(anat_dir, f"{subject}_brain_mask.nii.gz")
    if os.path.exists(mask_file):
        mask_img = image.load_img(mask_file)
    else:
        print(f"No brain mask found for {subject}, computing one using compute_epi_mask...")
        mask_img = compute_epi_mask(fmri_img)
    
    # Extract the time-series from the fMRI image using the mask
    time_series = masking.apply_mask(fmri_img, mask_img)  # shape: (n_timepoints, n_voxels)
    
    # Compute the mean signal across all voxels for each time point (whole-brain average)
    mean_time_series = time_series.mean(axis=1)  # shape: (n_timepoints,)
    
    # Define a fixed sequence length (e.g., 100 time points) as expected by the model
    SEQUENCE_LENGTH = 100
    if len(mean_time_series) < SEQUENCE_LENGTH:
        mean_time_series = np.pad(mean_time_series, (0, SEQUENCE_LENGTH - len(mean_time_series)), 'constant')
    else:
        mean_time_series = mean_time_series[:SEQUENCE_LENGTH]
    
    # Save the resulting 1D time-series to a CSV file named by the subject ID
    df_ts = pd.DataFrame({'activity': mean_time_series})
    output_file = os.path.join(output_dir, f"{subject}.csv")
    df_ts.to_csv(output_file, index=False)
    print(f"Saved time-series for {subject} to {output_file}")
