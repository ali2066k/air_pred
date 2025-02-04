import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, CropForegroundd, ScaleIntensityRanged, Rotate90, Rotate90d, ToTensord,
    AsDiscrete, Compose
)
from monai.data import Dataset, DataLoader, decollate_batch
from utils import get_model  # Ensure this function is available for loading your model

# Set up inference parameters
MODEL_CHECKPOINT = "/content/models/bel_old/best_metric_awc_64-0.87.ckpt"  # Update with your model path
INPUT_IMAGE_DIR = "/content/data_samples/imagesTr/"  # Directory containing NIfTI images
INPUT_LUNG_DIR = "/content/data_samples/lungsTr/"  # Directory containing NIfTI lung masks
OUTPUT_DIR = "/content/data_samples/predsTr/"  # Where to save results
PATCH_SIZE = (256, 256, 256)  # Update as needed
BATCH_SIZE = 1  # Adjust based on available memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params_dict = {
    'PATCH_SIZE': (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    'BATCH_SIZE': 1,
    'MAX_CARDINALITY': 120,
    'NUM_WORKERS': 0,
    'PIN_MEMORY': True,
    'AVAILABLE_GPUs': torch.cuda.device_count(),
}
params={"MODEL_NAME": "AttentionUNet",
        "IN_CHANNELS": 1,
        "OUT_CHANNELS": 1,
        "CHANNELS": 16,
        "N_LAYERS": 5,
        "STRIDES": 2,
        "SPATIAL_DIMS": 3,
        "DROPOUT": 0.0
        }
# Load the trained model
print("Loading model...")
model = get_model(params)  # Adjust params
print(model)
# Load the checkpoint
checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)

# Modify the state_dict to remove "_model." prefix from keys
new_state_dict = {}
for key, value in checkpoint["state_dict"].items():
    new_key = key.replace("_model.", "")  # Remove the prefix
    new_state_dict[new_key] = value

# Load the modified state_dict into the model
model.load_state_dict(new_state_dict, strict=False)
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
post_pred = Compose([
            AsDiscrete(threshold=0.5),
            Rotate90(k=1, spatial_axes=(0, 1))
        ])
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
        tst_outputs = [post_pred(i) for i in decollate_batch(output_tensor)]
        tst_filename = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
        affine = batch['image_meta_dict']['affine'][0]
        spt_size = batch['image_meta_dict']['spatial_shape'][0].tolist()
        x1, x2, x3 = batch['foreground_start_coord'][0][0], batch['foreground_start_coord'][0][1], \
            batch['foreground_start_coord'][0][2]
        y1, y2, y3 = batch['foreground_end_coord'][0][0], batch['foreground_end_coord'][0][1], \
            batch['foreground_end_coord'][0][2]

        output_corr = np.zeros(spt_size)
        label_corr = np.zeros(spt_size)

        output_corr[x1:y1, x2:y2, x3:y3] = np.squeeze(tst_outputs[0].type(torch.int64).cpu().numpy())


        # Load original NIfTI to keep affine and header
        original_nifti = nib.load(batch["image_meta_dict"]["filename_or_obj"][0])

        # Save the predicted segmentation
        output_nifti = nib.Nifti1Image(output_corr, affine=affine, header=original_nifti.header)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        nib.save(output_nifti, output_path)

        print(f"Saved: {output_path}")

print("Inference completed.")
