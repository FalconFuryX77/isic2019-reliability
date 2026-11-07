"""
Explainability — Grad-CAM and Grad-CAM++.
=========================================
Generates saliency heatmaps for selected test images across four cases:
  high-confidence correct, high-confidence incorrect,
  low-confidence correct, low-confidence incorrect.

Reference: Selvaraju et al. 2017 (Grad-CAM), Chattopadhay et al. 2018 (Grad-CAM++).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.builders import get_gradcam_target_layer


def gradcam(model, input_tensor, target_layer, target_class=None, method="gradcam"):
    """Compute Grad-CAM / Grad-CAM++ heatmap for a single input image.

    input_tensor: (1, 3, H, W) on device.
    Returns (heatmap, logits) where heatmap is (H, W) in [0, 1].
    """
    model.eval()
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    input_tensor = input_tensor.clone().requires_grad_(True)
    logits = model(input_tensor)

    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()
    score = logits[0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations["value"][0]          # (C, H', W')
    grad = gradients["value"][0]           # (C, H', W')

    handle_f.remove()
    handle_b.remove()

    if method == "gradcam":
        weights = grad.mean(dim=(1, 2), keepdim=True)      # (C, 1, 1)
        cam = F.relu((weights * act).sum(dim=0))           # (H', W')
    elif method == "gradcampp":
        # Grad-CAM++ weighting (positive gradients only)
        pos_grad = F.relu(grad)
        eps = 1e-10
        alpha_num = grad * grad
        alpha_den = 2 * grad * grad + (act * pos_grad).sum(dim=(1, 2), keepdim=True) + eps
        alpha = alpha_num / alpha_den                     # (C, H', W')
        weights = (pos_grad * alpha).sum(dim=(1, 2))      # (C,)
        cam = F.relu((weights.view(-1, 1, 1) * act).sum(dim=0))
    else:
        raise ValueError(method)

    # Normalize to [0, 1] and resize to input size
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
    heatmap = np.uint8(255 * cam)
    from PIL import Image
    heatmap = np.array(Image.fromarray(heatmap).resize(
        (input_tensor.shape[-1], input_tensor.shape[-2]), Image.BILINEAR)).astype(np.float32) / 255.0

    return heatmap, logits.detach()


def overlay_heatmap(image_np, heatmap, alpha=0.4, cmap="jet"):
    """Overlay a normalized heatmap on a uint8 RGB image array (H, W, 3)."""
    import matplotlib
    matplotlib.use("Agg")
    colored = matplotlib.colormaps[cmap](heatmap)[:, :, :3]   # (H, W, 3) in [0,1]
    overlay = (1 - alpha) * image_np / 255.0 + alpha * colored
    return np.clip(overlay, 0, 1)


def _load_image_tensor(path, image_size):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    return t(img).unsqueeze(0), img


def generate_explainability(cfg, model, loader, class_names, device, exp_dir,
                            num_per_category=3, method="gradcam"):
    """Select representative test images and generate heatmaps.

    Categories: correct-high, correct-low, incorrect-high, incorrect-low
    (confidence from the deterministic softmax).
    """
    from src.evaluation.evaluate import get_predictions
    from src.data.dataset import make_transforms, IMAGENET_MEAN, IMAGENET_STD

    exp_dir = Path(exp_dir)
    cam_dir = exp_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    probs, y_true = get_predictions(model, loader, device)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (y_pred == y_true).astype(bool)

    dataset = loader.dataset
    target_layer = get_gradcam_target_layer(model)
    norm = make_transforms(cfg.preprocessing.image_size, augment=False,
                           mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Bucket samples by category
    buckets = {
        "high_conf_correct": np.where(correct & (conf >= 0.85))[0],
        "high_conf_incorrect": np.where(~correct & (conf >= 0.85))[0],
        "low_conf_correct": np.where(correct & (conf < 0.60))[0],
        "low_conf_incorrect": np.where(~correct & (conf < 0.60))[0],
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    summary = {}
    for cat_name, idxs in buckets.items():
        idxs = idxs[:num_per_category]
        summary[cat_name] = {"n_available": int(len(idxs)), "examples": []}
        for i, idx in enumerate(idxs):
            img_path = dataset.paths[idx]
            true_c = class_names[y_true[idx]]
            pred_c = class_names[y_pred[idx]]

            # Deterministic preprocessed tensor for the model
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = norm(pil_img).unsqueeze(0).to(device)

            heatmap, logits = gradcam(model, input_tensor, target_layer,
                                      target_class=int(y_pred[idx]), method=method)

            # Original image resized for display
            from torchvision import transforms
            resized = np.array(pil_img.resize(
                (cfg.preprocessing.image_size, cfg.preprocessing.image_size),
                Image.BICUBIC)).astype(np.float32)
            overlay = overlay_heatmap(resized, heatmap)

            fname = f"{cat_name}_{i}_idx{idx}_true{true_c}_pred{pred_c}.png"
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(resized.astype(np.uint8)); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM heatmap"); axes[1].axis("off")
            axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
            fig.suptitle(f"True: {true_c} | Pred: {pred_c} | conf: {conf[idx]:.3f} | {cat_name}", fontsize=10)
            plt.tight_layout()
            plt.savefig(cam_dir / fname, dpi=140)
            plt.close()

            summary[cat_name]["examples"].append({
                "image": img_path,
                "true_label": true_c,
                "predicted_label": pred_c,
                "confidence": float(conf[idx]),
                "file": fname,
            })

    from src.utils import save_json
    save_json(summary, exp_dir / "gradcam_summary.json")
    print(f"\n  Grad-CAM examples saved to: {cam_dir}")
    return summary