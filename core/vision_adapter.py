"""Optional BigEarthNet-19 image classification adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


BIGEARTHNET_LABELS = [
    "Agro-forestry areas",
    "Arable land",
    "Bare areas",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Permanent crops",
    "Pastures",
    "Peatbogs",
    "Rice fields",
    "Rocks and stones",
    "Salt marshes",
    "Transitional woodland/shrub",
]


class BigEarthNetVisionAdapter:
    """Run optional BigEarthNet-19 predictions with a local ResNet-50 model.

    The adapter is deliberately optional: environments without ``torch`` or
    ``timm`` remain usable and return an empty prediction dictionary.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        threshold: float = 0.35,
        top_k: int = 5,
    ) -> None:
        self.threshold = float(threshold)
        self.top_k = max(1, int(top_k))
        self.model = None
        self.torch = None
        self.available = False

        try:
            import timm
            import torch
        except ImportError:
            return

        try:
            path = Path(weights_path) if weights_path else (
                Path(__file__).resolve().parent.parent
                / "weights" / "resnet50_pretrained.pth"
            )
            model = timm.create_model(
                "resnet50", pretrained=False, num_classes=1000
            )
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(
                checkpoint, dict
            ) else checkpoint
            if isinstance(state_dict, dict):
                state_dict = {
                    key.removeprefix("module."): value
                    for key, value in state_dict.items()
                }
                model.load_state_dict(state_dict, strict=False)
            model.reset_classifier(19)
            model.eval()
            self.model = model
            self.torch = torch
            self.available = True
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            # Missing, incompatible, or corrupt optional model assets should
            # never prevent the rest of the application from starting.
            print(e)
            return

    def predict_labels(self, image_np: np.ndarray) -> Dict[str, float]:
        """Return confident BigEarthNet labels mapped to sigmoid scores."""
        if not self.available or self.model is None or self.torch is None:
            return {}

        try:
            image = np.asarray(image_np)
            if image.ndim != 3:
                return {}
            if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] < 3:
                return {}
            image = image[..., :3].astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0
            image = np.clip(image, 0.0, 1.0)

            tensor = self.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            tensor = self.torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
            mean = self.torch.tensor(
                [0.485, 0.456, 0.406], dtype=tensor.dtype
            ).view(1, 3, 1, 1)
            std = self.torch.tensor(
                [0.229, 0.224, 0.225], dtype=tensor.dtype
            ).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std

            with self.torch.no_grad():
                logits = self.model(tensor)
                scores = self.torch.sigmoid(logits).flatten().cpu().numpy()
            ranked = np.argsort(scores)[::-1]
            return {
                BIGEARTHNET_LABELS[index]: round(float(scores[index]), 4)
                for index in ranked[: self.top_k]
                if scores[index] >= self.threshold
            }
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return {}