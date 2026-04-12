"""
dataloaders/yolo_tap_dataset.py

Detic-compatible labelled dataset built from:
  - YOLOWorld         for open-vocabulary box detection
  - Tokenize Anything for box-prompted mask prediction
  - CLIP              for per-instance image embeddings

The returned samples intentionally match DeticDenseLabelledDataset so the
training code can reuse the same downstream path.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import einops
import numpy as np
import torch
import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).parent.parent
_CLIP_REPO = _REPO_ROOT / "CLIP"
if _CLIP_REPO.exists() and str(_CLIP_REPO) not in sys.path:
    sys.path.insert(0, str(_CLIP_REPO))

import clip

YOLO_WORLD_MODEL_DEFAULT = str(
    _REPO_ROOT / "checkpoints" / "yolo" / "yolov8s-world.pt"
)
TAP_CHECKPOINT_DEFAULT = str(
    _REPO_ROOT / "checkpoints" / "tap" / "tap_vit_b_v1_1.pkl"
)
TAP_MODEL_TYPE_DEFAULT = "tap_vit_b"
TAP_IMAGE_SIZE_DEFAULT = 1024


def _ensure_ultralytics_clip_importable():
    """YOLOWorld depends on a top-level `clip` import."""
    clip_repo = _REPO_ROOT / "CLIP"
    clip_repo_str = str(clip_repo)
    if clip_repo.exists() and clip_repo_str not in sys.path:
        sys.path.insert(0, clip_repo_str)


def _build_yolo_world(weights_path: str, classes: List[str]):
    _ensure_ultralytics_clip_importable()
    from ultralytics import YOLOWorld

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLOWorld weights not found: {weights}. "
            "Set --yolo-world-model to a local checkpoint path."
        )

    model = YOLOWorld(str(weights))
    model.set_classes(classes)
    return model


def _parse_tap_device_index(device: str) -> int:
    if isinstance(device, str) and device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def _build_tap(model_type: str, checkpoint: str, device: str, image_size: int):
    from tokenize_anything import model_registry

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Tokenize Anything checkpoint not found: {ckpt}. "
            "Set --tap-checkpoint to a local checkpoint path."
        )
    if model_type not in model_registry:
        raise ValueError(
            f"Unknown TAP model type '{model_type}'. "
            f"Available: {sorted(model_registry.keys())}"
        )

    use_cuda = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()
    tap_dtype = "float16" if use_cuda else "float32"
    model = model_registry[model_type](
        checkpoint=str(ckpt),
        device=_parse_tap_device_index(device),
        dtype=tap_dtype,
        image_size=image_size,
    )
    if not use_cuda:
        model = model.float().to(torch.device("cpu"))
    return model


def _resize_image_and_boxes_to_square(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    target_size: int,
):
    h, w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scale_x = target_size / float(w)
    scale_y = target_size / float(h)
    resized_boxes = boxes_xyxy.copy().astype(np.float32)
    resized_boxes[:, [0, 2]] *= scale_x
    resized_boxes[:, [1, 3]] *= scale_y
    return resized, resized_boxes


class YoloTAPLabelledDataset(Dataset):
    """
    Generates a dense 3-D labelled dataset using YOLOWorld + Tokenize Anything.

    The output schema matches DeticDenseLabelledDataset / GDinoSAM2LabelledDataset.
    """

    def __init__(
        self,
        view_dataset,
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name: str = "all-mpnet-base-v2",
        device: str = "cuda",
        box_threshold: float = 0.1,
        subsample_prob: float = 0.2,
        yolo_world_model: str = YOLO_WORLD_MODEL_DEFAULT,
        tap_checkpoint: str = TAP_CHECKPOINT_DEFAULT,
        tap_model_type: str = TAP_MODEL_TYPE_DEFAULT,
        tap_image_size: int = TAP_IMAGE_SIZE_DEFAULT,
        visualize_results: bool = False,
        visualization_path: Optional[str] = None,
        # ---- compat kwargs (ignored or aliased) ----
        detic_threshold: float = 0.3,
        use_lseg: bool = False,
        use_extra_classes: bool = False,
        use_gt_classes: bool = True,
        exclude_gt_images: bool = False,
        gt_inst_images=None,
        gt_sem_images=None,
        use_scannet_colors: bool = True,
        num_images_to_label: int = -1,
        batch_size: int = 1,
    ):
        view_data = (
            view_dataset.dataset
            if isinstance(view_dataset, Subset)
            else view_dataset
        )
        self._image_width, self._image_height = view_data.image_size
        self._valid_min_depth = getattr(view_data, "min_depth", 0.0)
        self._valid_max_depth = getattr(view_data, "max_depth", 3.0)
        self._device = device
        self._subsample_prob = subsample_prob
        self._tap_image_size = tap_image_size

        # Keep the familiar CLI threshold behavior; a lower explicit Detic threshold
        # still acts as a generic detection threshold override.
        self._box_threshold = box_threshold if detic_threshold >= 0.3 else min(box_threshold, detic_threshold)

        self._all_classes = [
            self._process_text(x) for x in view_data._id_to_name.values()
        ]
        logger.info(
            "YOLOWorld classes (%d): %s",
            len(self._all_classes),
            ", ".join(self._all_classes),
        )

        self._visualize = visualize_results
        if self._visualize:
            assert visualization_path is not None, (
                "visualization_path must be set when visualize_results=True"
            )
            self._visualization_path = Path(visualization_path)
            os.makedirs(self._visualization_path, exist_ok=True)

        clip_model, self._clip_preprocess = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_encoding_model_name)

        self._label_xyz: List[torch.Tensor] = []
        self._label_rgb: List[torch.Tensor] = []
        self._label_weight: List[torch.Tensor] = []
        self._label_idx: List[torch.Tensor] = []
        self._text_ids: List[torch.Tensor] = []
        self._image_features: List[torch.Tensor] = []
        self._distance: List[torch.Tensor] = []
        self._text_id_to_feature = {}

        logger.info("Loading YOLOWorld …")
        yolo_world = _build_yolo_world(yolo_world_model, self._all_classes)
        logger.info("Loading Tokenize Anything …")
        tap = _build_tap(tap_model_type, tap_checkpoint, device, tap_image_size)

        self._run_labeling(view_dataset, yolo_world, tap, clip_model)

        del yolo_world
        del tap
        del clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        text_strings = [self._process_text(x) for x in self._all_classes]
        text_strings += self._all_classes
        with torch.no_grad():
            embedded = sentence_model.encode(text_strings)
            embedded = torch.from_numpy(embedded).float()
        del sentence_model

        for i, feat in enumerate(embedded):
            self._text_id_to_feature[i] = feat

        if not self._label_xyz:
            raise RuntimeError(
                "YoloTAPLabelledDataset produced no labelled points. "
                "Check YOLO weights, TAP checkpoint, class names, and depth validity."
            )

        self._label_xyz = torch.cat(self._label_xyz).float()
        self._label_rgb = torch.cat(self._label_rgb).float()
        self._label_weight = torch.cat(self._label_weight).float()
        self._image_features = torch.cat(self._image_features).float()
        self._text_ids = torch.cat(self._text_ids).long()
        self._label_idx = torch.cat(self._label_idx).long()
        self._distance = torch.cat(self._distance).float()
        self._instance = torch.full_like(self._text_ids, -1).long()

        logger.info("YoloTAPLabelledDataset: %d labelled points", len(self._label_xyz))

    @torch.no_grad()
    def _run_labeling(self, dataset, yolo_world, tap, clip_model):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        tap_dtype = torch.float16 if str(self._device).startswith("cuda") and torch.cuda.is_available() else torch.float32
        label_idx = 0

        for frame_idx, data_dict in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset), desc="YOLO+TAP labeling"
        ):
            rgb_bhwc = data_dict["rgb"][..., :3]
            xyz_batch = data_dict["xyz_position"]

            for image_chw, coordinates in zip(
                einops.rearrange(rgb_bhwc, "b h w c -> b c h w"),
                xyz_batch,
            ):
                h, w = image_chw.shape[1], image_chw.shape[2]
                image_rgb = einops.rearrange(image_chw, "c h w -> h w c").numpy()
                if image_rgb.max() <= 1.0:
                    image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
                else:
                    image_rgb = image_rgb.clip(0, 255).astype(np.uint8)
                image_bgr = image_rgb[:, :, ::-1].copy()

                yolo_results = yolo_world.predict(
                    source=image_bgr,
                    conf=self._box_threshold,
                    verbose=False,
                    device=self._device,
                )
                result = yolo_results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)

                tap_image, tap_boxes = _resize_image_and_boxes_to_square(
                    image_bgr, boxes_xyxy, self._tap_image_size
                )
                tap_inputs = tap.get_inputs(
                    {"img": tap_image[None]},
                    dtype=tap_dtype if tap_dtype != torch.float32 else None,
                )
                tap_inputs.update(tap.get_features(tap_inputs))
                tap_inputs["boxes"] = tap_boxes
                tap_outputs = tap.get_outputs(tap_inputs)

                iou_pred = tap_outputs["iou_pred"]
                mask_pred = tap_outputs["mask_pred"]
                best_idx = iou_pred.argmax(dim=1)
                batch_indices = torch.arange(mask_pred.shape[0], device=mask_pred.device)
                best_masks = mask_pred[batch_indices, best_idx]
                best_masks = tap.upscale_masks(best_masks.unsqueeze(1), (h, w)).squeeze(1)
                best_masks = best_masks > 0
                best_masks = best_masks.detach().cpu()

                reshaped_coords, valid_mask = self._get_valid_coords(
                    coordinates, data_dict
                )
                reshaped_rgb = torch.tensor(image_rgb)

                for class_idx, score, pred_mask, box_xyxy in zip(
                    class_ids, scores, best_masks, boxes_xyxy
                ):
                    if class_idx < 0 or class_idx >= len(self._all_classes):
                        continue

                    mask_np = pred_mask.numpy().astype(np.uint8)
                    mask_np = cv2.erode(mask_np, np.ones((3, 3), np.uint8), iterations=1)
                    pred_mask = torch.from_numpy(mask_np).bool()

                    real_mask = pred_mask[valid_mask]
                    real_mask_rect = valid_mask & pred_mask
                    total_pts = int(real_mask.sum().item())
                    if total_pts == 0:
                        continue

                    keep = torch.rand(total_pts) < self._subsample_prob
                    if not keep.any():
                        continue
                    n_kept = int(keep.sum().item())

                    x1, y1, x2, y2 = box_xyxy.astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w), min(y2, h)
                    crop = image_rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    clip_input = self._clip_preprocess(Image.fromarray(crop)).unsqueeze(0).to(self._device)
                    clip_feat = clip_model.encode_image(clip_input).squeeze(0).cpu().float()

                    self._label_xyz.append(reshaped_coords[real_mask][keep])
                    self._label_rgb.append(reshaped_rgb[real_mask_rect][keep].float())
                    self._text_ids.append(
                        torch.full((n_kept,), int(class_idx), dtype=torch.float32)
                    )
                    self._label_weight.append(
                        torch.full((n_kept,), float(score))
                    )
                    self._image_features.append(
                        einops.repeat(clip_feat, "d -> n d", n=n_kept)
                    )
                    self._label_idx.append(
                        torch.full((n_kept,), float(label_idx))
                    )
                    self._distance.append(torch.zeros(n_kept))
                    label_idx += 1

                    if self._visualize:
                        self._save_debug(
                            image_rgb,
                            pred_mask,
                            self._all_classes[int(class_idx)],
                            frame_idx,
                        )

    def _get_valid_coords(self, coordinates, data_dict):
        if "conf" in data_dict:
            valid_mask = (
                torch.as_tensor(
                    (~np.isnan(data_dict["depth"].numpy()))
                    & (data_dict["conf"].numpy() == 2)
                    & (data_dict["depth"].numpy() > self._valid_min_depth)
                    & (data_dict["depth"].numpy() < self._valid_max_depth)
                )
                .squeeze(0)
                .bool()
            )
            reshaped_coords = torch.as_tensor(coordinates)
            return reshaped_coords, valid_mask

        reshaped_coords = einops.rearrange(coordinates, "c h w -> (h w) c")
        valid_mask = torch.ones(
            coordinates.shape[1], coordinates.shape[2], dtype=torch.bool
        )
        return reshaped_coords, valid_mask

    def _save_debug(self, image_rgb, pred_mask, class_name, frame_idx):
        vis = image_rgb.copy()
        overlay = vis.copy()
        overlay[pred_mask.numpy().astype(bool)] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        out_path = self._visualization_path / f"{frame_idx:04d}_{class_name}.jpg"
        cv2.imwrite(str(out_path), vis[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])

    @staticmethod
    def _process_text(x: str) -> str:
        return x.replace("-", " ").replace("_", " ").strip().lower()

    process_text = _process_text

    def __len__(self):
        return len(self._label_xyz)

    def __getitem__(self, idx):
        return {
            "xyz": self._label_xyz[idx].float(),
            "rgb": self._label_rgb[idx].float(),
            "label": self._text_ids[idx].long(),
            "instance": self._instance[idx].long(),
            "img_idx": self._label_idx[idx].long(),
            "distance": self._distance[idx].float(),
            "clip_vector": self._text_id_to_feature[
                self._text_ids[idx].item()
            ].float(),
            "clip_image_vector": self._image_features[idx].float(),
            "semantic_weight": self._label_weight[idx].float(),
        }
