import torch
import torchvision
import cv2
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np
import os

# -----------------------------
# 1. Build model
# -----------------------------
# The checkpoint was actually trained with regnet_y_32gf (based on state dict keys)
model = torchvision.models.regnet_y_32gf(weights=None)
model.fc = torch.nn.Linear(3712, 2526)
device = torch.device("cpu")
model.to(device)

# -----------------------------
# 2. Load checkpoint
# -----------------------------
ckpt_path = r"C:\Users\rvc60\Insect-Classifier\weights\model.pth"
print(f"Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

state_dict = ckpt["model"]  # only model available
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)

# Disable BatchNorm running stats for stable inference
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats = False

model.eval()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Additional debugging info
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Model in eval mode: {not model.training}")

# -----------------------------
# 3. Image preprocessing
# -----------------------------
def preprocess_image(image):
    # Match exactly the original evaluate.py transforms_validation function
    crop_size = 224
    resize_size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR
    transforms_val = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),  # No antialias=True in original
        transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.fromarray(np.uint8(image))
    image = transforms_val(image).reshape((1, 3, 224, 224))  # Match original reshaping
    return image

# -----------------------------
# 4. Load and run inference
# -----------------------------
img_path = r"C:\Users\rvc60\Insect-Classifier\OSK.jpg"
image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
input_tensor = preprocess_image(image)  # Don't move to device yet, match original

# Match original evaluate.py inference exactly
with torch.inference_mode():
    input_tensor = input_tensor.to(device, non_blocking=True)  # Move to device here like original
    output = model(input_tensor)
    op = torch.nn.functional.softmax(output, dim=1)
    op_ix = torch.argmax(op)  # Use argmax like original
    
confidence = op[0][op_ix].item()
pred_index = op_ix.item()

# -----------------------------
# 5. Map to labels
# -----------------------------
df = pd.read_csv(r"C:\Users\rvc60\Insect-Classifier\classes.csv")
scientific_names = list(df["genus"] + " " + df["species"])
roles = list(df["Role in Ecosystem"])

print("\n===== Inference Result =====")
# Match original evaluate.py confidence threshold (0.97)
if confidence >= 0.97:
    print(f"Scientific Name: {scientific_names[pred_index]}")
    print(f"Role in Ecosystem: {roles[pred_index]}")
else:
    print(f"Maybe OOD. Scientific Name: {scientific_names[pred_index]}")
    print(f"Role in Ecosystem: {roles[pred_index]}")

print(f"Prediction Index: {pred_index}")
print(f"Confidence: {confidence:.4f}")

# Show top 5 predictions for debugging
print("\n===== Top Prediction =====")
top_prob, top_idx = op.topk(1)
print(f"{scientific_names[top_idx.item()]} (conf: {top_prob.item():.4f})")
