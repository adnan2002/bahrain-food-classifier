# model.pkl — Investigation Report

## 1. What model.pkl is

`model.pkl` is a **PyTorch state dict** (only weights, no training state) of a **NoisyViT-B/16** — a ViT-Base Vision Transformer classifier.

| Property | Value |
|---|---|
| Architecture | ViT-B/16: 12 transformer blocks, hidden dim 768, 12 heads, MLP 3072, patch 16×16 |
| Input | 224×224 RGB image |
| Parameters | **85,983,985 (~86M)** |
| Size | 344 MB (float32 — the file is almost entirely weights) |
| Training | Pretrained on ImageNet, then fine-tuned on **CNFOOD-241** (Chinese food, 241 classes) |
| Head | 241 output classes (Chinese dishes) |
| Source | "Improving Food Image Recognition with Noisy Vision Transformer" (Tonmoy-Ghosh/NoisyViT_Food, arXiv:2503.18997) |
| Provenance evidence | Checkpoint name `acc_0.9656_lr_1e-05_bs_16_layer_11_base_224_16_linear_ImageNet_NoisyViT` matches the paper's training config: 96.56% top-1 on CNFOOD-241, lr 1e-5, batch 16, last 11 layers fine-tuned, "linear" noise type |

The checkpoint loads **1:1** into timm's `vit_base_patch16_224` (zero missing / zero unexpected keys), so it can be treated as a standard pretrained ViT-B/16.

## 2. Tests I ran

Environment: `.venv` with torch 2.13 (CPU), timm 1.0.28, torchvision 0.28.

### 2.1 Structure inspection (no torch needed)
- Confirmed the file is a `torch.save` zip archive; decoded the legacy pickle format manually with numpy to verify the tensor layout and count parameters **before** loading it in torch (85,983,985 total, all non-zero — no dead weights).

### 2.2 Load test
- Loaded the state dict into `timm.create_model('vit_base_patch16_224', num_classes=241)` — all 152 tensors matched key-for-key.
- Ran real inference on your food images: ~150 ms/image on CPU (faster batched / GPU).

### 2.3 Prediction sanity check (241-class Chinese-food head)
- The head cannot name Bahraini dishes (it only knows Chinese foods), but predictions are stable and cluster per class — e.g. eggs_tomato images → class #50 at up to 95% confidence, karak → class #205. Weights are meaningful.

### 2.4 Feature-quality test (the important one)
- Froze the backbone, extracted the 768-d CLS features (pre-head) from **all 270 usable images** (9 classes × 30).
- Leave-one-out **nearest-centroid** classification with the frozen backbone:

```
Overall: 97.4%  (263/270)
Per class: balaleet 100%, eggs_tomato 100%, fish 100%, harees 100%,
           karak 100%, ma3krona 100%, nakhaj 100%, tikka 100%, liver 96.7%
```

- Zero training was done — this is the pure pretrained feature extractor on your data. The food-specialized features already separate your classes almost perfectly.

## 3. Transfer learning for your dataset

Your setup: 12 classes (balaleet, eggs_tomato, fish, gaimat, halwa, harees, karak, liver, ma3krona, nakhaj, samboosa, tikka), ~30 images per class, **3 classes empty (gaimat, halwa, samboosa)**.

### 3.1 What to do

1. **Replace the head**: swap the 241-class head for a 12-class linear layer (`head = nn.Linear(768, 12)`). The Chinese-food head is useless for your classes.
2. **Freeze the backbone**, train only the head (linear probe) — this is where the 97.4% test says you already have a working model; training the probe should push it higher and properly calibrate it.
3. **Optionally unfreeze the last 2–4 blocks** at a small LR (~1e-5) once the probe converges — this mirrors exactly how the paper's own 96.56% model was trained (last 11 layers, lr 1e-5).
4. **Use the paper's augmentation stack**: RandomResizedCrop(scale 0.05–1.0), RandomHorizontalFlip, RandomRotation(10), RandAugment.
5. **Preprocessing**: `Resize((224,224))` + ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) — verified from the paper's dataloader.

### 3.2 Constraints to respect

- **~86M parameters vs ~270 images**: do not fine-tune the full model end-to-end on this dataset — it will overfit. Frozen backbone + head is the safe route; partial unfreezing (2–4 blocks) with strong augmentation is the max you should attempt before collecting more data.
- **3 empty classes must be filled** before any training covers all 12 classes.
- The model is a **classifier** (tells you which food is in the image), not a detector (does not draw boxes around food).

### 3.3 Expected outcome

With frozen features you already get ~97% nearest-centroid separation; a trained linear probe + light fine-tuning on top should give a strong baseline for a 12-class food classifier with ~30 images per class.

## 4. Important: it is NOT a YOLO model

You mentioned using it "as a YOLO model" — this is not possible directly:

- `model.pkl` is a **Vision Transformer classifier** (takes an image, outputs class probabilities over 241 categories).
- **YOLO** (v8/v11) is a **CNN-based object detector** (takes an image, outputs bounding boxes + class labels). It is a completely different architecture with a different file format (`.pt`/`.engine`), and it is trained on **bounding-box annotations**, not class folders.
- A ViT classifier cannot be loaded into or converted to YOLO weights — the layers, tensor shapes, and output semantics do not match.

### What you can do instead

| Goal | Right approach |
|---|---|
| Classify food in a single image (what's this dish?) | Use this ViT via transfer learning (section 3) |
| Detect + locate food (boxes around dishes) | Train **YOLOv8/v11** from scratch on your 12 classes with box annotations |
| Detect food, then classify the dish | Two-stage pipeline: YOLO detects & crops → this ViT classifies the crop (the best of both — YOLO handles localization, the ViT handles the fine-grained dish identity) |

If your goal is really object detection, keep this model only as the classifier stage — for the detector you need a YOLO pretrained checkpoint (e.g. `yolov8n.pt`) plus labeled bounding boxes for your 12 classes.

---

### Quick reference (commands/values)

- Params: `85,983,985` | fp32 size: `343.9 MB`
- Model constructor: `timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=241)` + `load_state_dict`
- Normalization: ImageNet mean/std, `Resize((224,224))`
- Provenance: CNFOOD-241 fine-tune, 96.56% top-1, paper arXiv:2503.18997
