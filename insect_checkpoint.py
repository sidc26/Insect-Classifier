import torch

ckpt_path = r"C:\Users\rvc60\Insect-Classifier\weights\model.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Handle different possible structures (model, model_ema, or flat dict)
if "model_ema" in ckpt:
    print("✅ Found EMA weights in checkpoint.")
    state_dict = ckpt["model_ema"]
elif "model" in ckpt:
    print("✅ Found 'model' weights in checkpoint.")
    state_dict = ckpt["model"]
else:
    print("⚠️ No 'model' or 'model_ema' key found; assuming flat state_dict.")
    state_dict = ckpt

print(f"\nTotal parameters in checkpoint: {len(state_dict)}")

# Show first few layer names
print("\nFirst 20 keys:")
for k in list(state_dict.keys())[:20]:
    print(" ", k)

# Check if this includes convolutional layers (backbone) or only classifier
has_conv = any("conv" in k or "stem" in k for k in state_dict.keys())
has_fc = any("fc" in k for k in state_dict.keys())

print("\nSummary:")
print(f"Contains convolutional backbone layers? {'✅ Yes' if has_conv else '❌ No'}")
print(f"Contains final classifier (fc) layer? {'✅ Yes' if has_fc else '❌ No'}")

print("Keys at top level of checkpoint:", list(ckpt.keys())[:5])

print("True" if "model_ema" in ckpt.keys() else "False")