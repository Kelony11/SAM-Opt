import time
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor


ROOT = Path(__file__).resolve().parents[1]

MODEL_TYPE = "vit_b"
CHECKPOINT = ROOT / "models/sam_vit_b_01ec64.pth"
IMAGE_PATH = ROOT / "data/raw/truck.jpg"
OUTPUT_DIR = ROOT / "results/baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Select available hardware
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Device: {device}")
print(f"Model: {MODEL_TYPE}")


# Load pretrained SAM
print("Loading SAM model...")

sam = sam_model_registry[MODEL_TYPE](
    checkpoint=str(CHECKPOINT)
)

sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)

print("SAM model loaded successfully")


# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Image shape: {image.shape}")


# Define a foreground point prompt
input_point = np.array([[500, 375]])
input_label = np.array([1])


# Synchronize GPU for accurate timing
def synchronize():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# Perform inference
synchronize()
start_time = time.perf_counter()

with torch.inference_mode():
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

synchronize()
inference_time = time.perf_counter() - start_time


# Print results
print(f"Masks generated: {len(masks)}")
print(f"Mask scores: {scores}")
print(f"Inference time: {inference_time:.4f} seconds")


# Select the highest-scoring mask
best_index = np.argmax(scores)
best_mask = masks[best_index]

print(f"Best predicted IoU: {scores[best_index]:.4f}")


# Save binary segmentation mask
cv2.imwrite(
    str(OUTPUT_DIR / "best_mask.png"),
    best_mask.astype(np.uint8) * 255
)


# Visualize original image and all candidate masks
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

for i, (mask, score) in enumerate(zip(masks, scores)):
    ax = axes[i + 1]

    ax.imshow(image)

    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [0.1, 0.6, 1.0, 0.5]

    ax.imshow(overlay)

    ax.scatter(
        input_point[:, 0],
        input_point[:, 1],
        color="lime",
        marker="*",
        s=200
    )

    ax.set_title(f"Mask {i + 1}: {score:.4f}")
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_segmentation.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Results saved to: {OUTPUT_DIR}")
print("Baseline inference completed successfully")

