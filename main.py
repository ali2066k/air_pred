import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, CropForegroundd, ScaleIntensityRanged, Rotate90d, ToTensord,
    AsDiscrete, Compose
)
from monai.data import Dataset, DataLoader, decollate_batch
from utils import get_model  # Ensure this function is available for loading your model

# Set up inference parameters
MODEL_CHECKPOINT = "/path/to/trained_model.ckpt"  # Update with your model path
INPUT_IMAGE_DIR = "/path/to/nifti_cases/images/"  # Directory containing NIfTI images
INPUT_LUNG_DIR = "/path/to/nifti_cases/lungs/"  # Directory containing NIfTI lung masks
OUTPUT_DIR = "/path/to/output_predictions/"  # Where to save results
PATCH_SIZE = (256, 256, 256)  # Update as needed
BATCH_SIZE = 1  # Adjust based on available memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
print("Loading model...")
model = get_model(params={"MODEL_NAME": "YourModel", "IN_CHANNELS": 1, "OUT_CHANNELS": 1})  # Adjust params
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE)["state_dict"])
model.to(DEVICE)
model.eval()

# Define preprocessing pipeline using Compose
preprocess = Compose([
    LoadImaged(keys=["image", "lung"], image_only=False),
    EnsureChannelFirstd(keys=["image", "lung"]),
    CropForegroundd(keys=["image", "lung"], source_key="lung", margin=[1, 1, 50], allow_smaller=True),
    ScaleIntensityRanged(
        keys="image",
        a_min=-1000,  # Adjust based on your dataset
        a_max=600,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    Rotate90d(keys=["image", "lung"], k=3),
    ToTensord(keys=["image", "lung"], dtype=torch.float32),
])

# Define post-processing pipeline
post_pred = Compose([AsDiscrete(threshold=0.5)])

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all NIfTI files
image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.endswith(".nii") or f.endswith(".nii.gz")]

# Create dataset for preprocessing
data_list = []
for file in image_files:
    image_path = os.path.join(INPUT_IMAGE_DIR, file)
    lung_path = os.path.join(INPUT_LUNG_DIR, file)  # Assumes lung mask has the same filename
    if os.path.exists(lung_path):
        data_list.append({"image": image_path, "lung": lung_path})
    else:
        print(f"Skipping {file}: Corresponding lung mask not found.")

# Create DataLoader
dataset = Dataset(data=data_list, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

print(f"Found {len(dataset)} NIfTI cases. Running inference...")

# Perform inference on each file
with torch.no_grad():
    for batch in dataloader:
        file_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])  # Extract filename

        # Move tensors to device
        input_tensor = batch["image"].to(DEVICE)

        # Run model inference
        output_tensor = sliding_window_inference(
            inputs=input_tensor, roi_size=PATCH_SIZE, sw_batch_size=BATCH_SIZE, predictor=model, overlap=0.25
        )

        # Apply post-processing
        output_tensor = [post_pred(i) for i in decollate_batch(output_tensor)][0]

        # Convert back to numpy
        output_array = output_tensor.cpu().numpy().squeeze()

        # Load original NIfTI to keep affine and header
        original_nifti = nib.load(batch["image_meta_dict"]["filename_or_obj"][0])

        # Save the predicted segmentation
        output_nifti = nib.Nifti1Image(output_array, affine=original_nifti.affine, header=original_nifti.header)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        nib.save(output_nifti, output_path)

        print(f"Saved: {output_path}")

print("Inference completed.")