"""Adapter for optional Grounding DINO model.

Provides a small, defensive wrapper that lazily loads GroundingDINO weights
if the library and weights are available. If the third-party library is not
installed, the adapter will gracefully return empty predictions and log a
warning rather than raising an import error.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class GroundingDINOAdapter:
    """Lazily-loadable GroundingDINO adapter.

    Parameters
    - weights: path to weights file (default: weights/groundingdino_swint_ogc.pth)

    Methods
    - predict_boxes(image_np, text_query) -> List[dict]: returns list of
      {'bbox': [x1,y1,x2,y2], 'score': float, 'label': str}
    """

    def __init__(self, weights: str | Path | None = None):
        self.weights = Path(weights) if weights is not None else Path("weights") / "groundingdino_swint_ogc.pth"
        self._model = None
        self._available = None
        self._loader_error = None

    def _ensure_model(self) -> bool:
        if self._available is not None:
            return self._available

        # Try to import libraries and construct a model. We don't assume a
        # specific repository layout; instead try a few common entrypoints
        # then fall back to a graceful disabled state.
        try:
            import torch  # type: ignore
        except Exception as e:
            logger.warning("Torch not available for GroundingDINO: %s", e)
            self._loader_error = e
            self._available = False
            return False

        if not self.weights.exists():
            logger.warning("GroundingDINO weights not found at %s", self.weights)
            self._available = False
            return False

        try:
            # Try to import common groundingdino loader functions; these will
            # vary between installations. We attempt a few likely names and
            # APIs to be tolerant.
            try:
                import groundingdino  # type: ignore
                # if the package exposes a convenience loader, use it
                if hasattr(groundingdino, "load_model"):
                    self._model = groundingdino.load_model(str(self.weights))
                elif hasattr(groundingdino, "build_model"):
                    # best-effort: call build_model then load state_dict
                    cfg = None
                    self._model = groundingdino.build_model(cfg)
                    if hasattr(self._model, "load_state_dict"):
                        sd = torch.load(str(self.weights), map_location="cpu")
                        self._model.load_state_dict(sd)
                else:
                    # fallback: try an inference helper module
                    from groundingdino.util import inference as _inf  # type: ignore
                    if hasattr(_inf, "GroundingDINO"):
                        self._model = _inf.GroundingDINO(str(self.weights))
                    else:
                        raise RuntimeError("No compatible GroundingDINO loader found in package")
            except Exception:
                # Attempt another common import path used by some forks/demos
                try:
                    from groundingdino.models import build_model as _build  # type: ignore
                    self._model = _build()
                    sd = torch.load(str(self.weights), map_location="cpu")
                    if hasattr(self._model, "load_state_dict"):
                        self._model.load_state_dict(sd)
                except Exception as e:
                    raise RuntimeError("Failed to build GroundingDINO model") from e

            # move model to cpu for safe inference unless user wishes otherwise
            try:
                if hasattr(self._model, "eval"):
                    self._model.eval()
            except Exception:
                pass

            self._available = True
            logger.info("GroundingDINO adapter loaded with weights %s", self.weights)
            return True
        except Exception as e:
            logger.warning("Could not initialize GroundingDINO adapter: %s", e)
            self._loader_error = e
            self._available = False
            return False

    def predict_boxes(self, image_np: np.ndarray, text_query: str) -> List[Dict[str, Any]]:
        """Run grounding on `image_np` for `text_query`.

        Returns a list of boxes (x1,y1,x2,y2) with scores and label text.
        If the model is unavailable, returns an empty list.
        """
        if not self._ensure_model():
            logger.warning("GroundingDINO not available; returning empty predictions")
            return []

        # Attempt to call a variety of possible inference entrypoints.
        preds = None
        try:
            # common: model.predict(image, caption) -> list/dict
            if hasattr(self._model, "predict"):
                preds = self._model.predict(image_np, text_query)
            elif hasattr(self._model, "inference"):
                preds = self._model.inference(image_np, [text_query])
            else:
                # try calling model directly
                preds = self._model(image_np, text_query)
        except Exception as e:
            logger.warning("GroundingDINO inference failed: %s", e)
            return []

        # Normalize output into list of dicts: {'bbox':[x1,y1,x2,y2],'score':f,'label':s}
        out: List[Dict[str, Any]] = []
        try:
            if isinstance(preds, dict):
                boxes = preds.get("boxes") or preds.get("pred_boxes")
                scores = preds.get("scores") or preds.get("confidences")
                labels = preds.get("labels") or [text_query] * (len(boxes) if boxes is not None else 0)
                if boxes is None:
                    return []
                boxes = np.asarray(boxes)
                scores = list(scores) if scores is not None else [1.0] * len(boxes)
                for b, s, l in zip(boxes.tolist(), scores, labels):
                    x1, y1, x2, y2 = map(float, b[:4])
                    out.append({"bbox": [x1, y1, x2, y2], "score": float(s), "label": str(l)})
                return out

            if isinstance(preds, (list, tuple)):
                for p in preds:
                    if isinstance(p, dict) and "bbox" in p:
                        out.append({"bbox": p["bbox"], "score": float(p.get("score", 1.0)), "label": p.get("label", text_query)})
                    else:
                        # assume array-like box
                        try:
                            x1, y1, x2, y2 = map(float, p[:4])
                            out.append({"bbox": [x1, y1, x2, y2], "score": 1.0, "label": text_query})
                        except Exception:
                            continue
                return out
        except Exception as e:
            logger.warning("Failed to normalize GroundingDINO output: %s", e)
            return []

        # unknown format
        return []
