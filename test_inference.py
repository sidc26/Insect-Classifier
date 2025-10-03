import torch
import torchvision
import cv2
from evaluate import evaluate

# -----------------------------
# 1. Build the model
# -----------------------------
model = torchvision.models.regnet_y_32gf(weights=None)  # no pretrained weights
model.fc = torch.nn.Linear(3712, 2526)  # match checkpoint output

# -----------------------------
# 2. Load checkpoint
# -----------------------------
ckpt_path = r"C:\Users\rvc60\Insect-Classifier\weights\model.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Extract only the model weights (ignore optimizer, lr_scheduler, etc.)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"⚠️ Missing keys: {missing}")
print(f"⚠️ Unexpected keys: {unexpected}")

# -----------------------------
# 3. Eval settings
# -----------------------------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()

# -----------------------------
# 4. Load test image
# -----------------------------
img_path = r"C:\Users\rvc60\Insect-Classifier\image.png"
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# 5. Run evaluation
# -----------------------------
result = evaluate(model, image)

print("\n===== Inference Result =====")
print(result)
