import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from .CLIP import clip
from .grid_hash_model import GridCLIPModel
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import open3d as o3d

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()


def _as_cpu_tensor(value):
    return torch.as_tensor(value).detach().cpu()


def _sorted_linear_weight_keys(state_dict):
    keys = [
        key for key, value in state_dict.items()
        if (
            key.startswith("_post_grid.trunk.")
            and key.endswith(".weight")
            and getattr(value, "ndim", 0) == 2
        )
    ]
    return sorted(keys, key=lambda key: int(key.split(".")[2]))


def _infer_model_kwargs_from_state_dict(state_dict, data, max_coords, min_coords):
    """Recover GridCLIPModel kwargs for checkpoints saved without metadata."""
    embeddings = state_dict["_grid_model.embeddings"]
    offsets = state_dict["_grid_model.offsets"]
    linear_weight_keys = _sorted_linear_weight_keys(state_dict)
    if not linear_weight_keys:
        raise ValueError("Checkpoint does not contain _post_grid linear weights.")

    image_rep_size = data[0]["clip_image_vector"].shape[-1]
    text_rep_size = data[0]["clip_vector"].shape[-1]
    output_dim = state_dict[linear_weight_keys[-1]].shape[0]
    expected_output_dim = image_rep_size + text_rep_size
    if output_dim != expected_output_dim:
        raise ValueError(
            "Checkpoint output dimension does not match the labelled dataset: "
            f"checkpoint={output_dim}, dataset={expected_output_dim}."
        )

    level_sizes = offsets[1:] - offsets[:-1]
    capped_level_size = int(level_sizes.max().item())
    log2_hashmap_size = int(np.log2(capped_level_size))
    if 2 ** log2_hashmap_size != capped_level_size:
        # If no level reached the cap, fall back to the GridCLIPModel default.
        log2_hashmap_size = 24

    return dict(
        image_rep_size=image_rep_size,
        text_rep_size=text_rep_size,
        mlp_depth=max(0, len(linear_weight_keys) - 1),
        mlp_width=state_dict[linear_weight_keys[0]].shape[0],
        log2_hashmap_size=log2_hashmap_size,
        num_levels=int(offsets.numel() - 1),
        level_dim=embeddings.shape[1],
        per_level_scale=2,
        max_coords=max_coords,
        min_coords=min_coords,
    )

@dataclass
class QueryResult:
    """Result of a single text query against the CLIP-Field."""
    query: str
    points: np.ndarray          # (N, 3) matched points above threshold
    best_point: np.ndarray      # (3,)   single highest-scoring point
    scores: np.ndarray          # (N,)   alignment scores for matched points
    best_score: float           # scalar score of best point


class CLIPFieldQuery:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 30_000,
        quantile_threshold: float = 0.9999,
        visual: bool = False,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.quantile_threshold = quantile_threshold
        self.visual = visual


        # load clip
        self._clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self._sentence_model = SentenceTransformer("all-mpnet-base-v2")
        self._clip_module = clip  # keep reference for tokenize()

        # load pointcloud
        self._data = torch.load(data_path, weights_only=False)
        self._all_xyz = self._data._label_xyz 
        max_coords, _ = self._all_xyz.max(dim=0)
        min_coords, _ = self._all_xyz.min(dim=0)

        # ---- Load trained CLIP-Field model ----
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_kwargs" in ckpt:
            model_kwargs = dict(ckpt["model_kwargs"])
            ckpt_max = _as_cpu_tensor(model_kwargs.get("max_coords", max_coords))
            ckpt_min = _as_cpu_tensor(model_kwargs.get("min_coords", min_coords))
            data_max = _as_cpu_tensor(max_coords)
            data_min = _as_cpu_tensor(min_coords)
            if not (
                torch.allclose(ckpt_max, data_max, atol=1e-4, rtol=1e-4)
                and torch.allclose(ckpt_min, data_min, atol=1e-4, rtol=1e-4)
            ):
                warnings.warn(
                    "Loaded checkpoint bounds differ from the labelled dataset. "
                    "The model will load, but query coordinates may be unreliable "
                    "unless the data and checkpoint belong to the same scene.",
                    RuntimeWarning,
                )
        else:
            model_kwargs = _infer_model_kwargs_from_state_dict(
                ckpt["model"], self._data, max_coords, min_coords
            )

        model_kwargs.setdefault(
            "image_rep_size", self._data[0]["clip_image_vector"].shape[-1]
        )
        model_kwargs.setdefault(
            "text_rep_size", self._data[0]["clip_vector"].shape[-1]
        )
        model_kwargs.setdefault("max_coords", max_coords)
        model_kwargs.setdefault("min_coords", min_coords)
        model_kwargs["device"] = str(self.device)

        self._label_model = GridCLIPModel(**model_kwargs).to(self.device)
        self._label_model.load_state_dict(ckpt["model"])
        self._label_model.eval()
        self._points_loader = DataLoader(
            self._all_xyz, batch_size=self.batch_size, num_workers=10,
        )

        print(f"[CLIPFieldQuery] Loaded {len(self._all_xyz)} scene points.")
        print(f"[CLIPFieldQuery] Model loaded from {model_path}")

    def query(
        self,
        text: str,
        quantile: Optional[float] = None,
        visual: Optional[bool] = None,
    ) -> QueryResult:
        results = self._run_queries(
            [text],
            quantile=quantile if quantile is not None else self.quantile_threshold,
            visual=visual if visual is not None else self.visual,
        )
        return results[0]

    def query_best(
        self,
        text: str,
        visual: Optional[bool] = None,
    ) -> np.ndarray:
        result = self.query(text, visual=visual)
        return result.best_point

    def query_batch(
        self,
        texts: List[str],
        quantile: Optional[float] = None,
        visual: Optional[bool] = None,
    ) -> Dict[str, QueryResult]:
        results = self._run_queries(
            texts,
            quantile=quantile if quantile is not None else self.quantile_threshold,
            visual=visual if visual is not None else self.visual,
        )
        return {r.query: r for r in results}

    def visualize_3d(
        self,
        results: Union[QueryResult, List[QueryResult], Dict[str, QueryResult]],
        save_path: Optional[str] = None,
        max_scene_points: int = 50000,
        best_point_radius: float = 0.03,
    ):
        if isinstance(results, dict):
            result_list = list(results.values())
        elif isinstance(results, QueryResult):
            result_list = [results]
        else:
            result_list = results

        all_xyz = self._all_xyz.detach().cpu().numpy()
        scene_step = max(1, len(all_xyz) // max_scene_points)
        scene_pts = all_xyz[::scene_step]

        colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                  "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

        def _hex_to_rgb(hex_color: str):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
        scene_pcd.paint_uniform_color((0.8, 0.8, 0.8))

        geometries = [scene_pcd]
        for i, r in enumerate(result_list):
            color = _hex_to_rgb(colors[i % len(colors)])
            if len(r.points) > 0:
                matched_pcd = o3d.geometry.PointCloud()
                matched_pcd.points = o3d.utility.Vector3dVector(r.points)
                matched_pcd.paint_uniform_color(color)
                geometries.append(matched_pcd)

            best_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=best_point_radius)
            best_sphere.translate(r.best_point)
            best_sphere.paint_uniform_color(color)
            best_sphere.compute_vertex_normals()
            geometries.append(best_sphere)

            ring_radius = best_point_radius * 6
            circle_segments = 24
            circle_points = []
            circle_lines = []
            for j in range(circle_segments):
                theta = 2 * np.pi * j / circle_segments
                px = r.best_point[0] + ring_radius * np.cos(theta)
                py = r.best_point[1] + ring_radius * np.sin(theta)
                pz = r.best_point[2]
                circle_points.append([px, py, pz])
                circle_lines.append([j, (j + 1) % circle_segments])

            circle_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(circle_points),
                lines=o3d.utility.Vector2iVector(circle_lines),
            )
            circle_line_set.colors = o3d.utility.Vector3dVector([color for _ in circle_lines])
            geometries.append(circle_line_set)

            star_offsets = [
                [ring_radius, 0.0, 0.0],
                [-ring_radius, 0.0, 0.0],
                [0.0, ring_radius, 0.0],
                [0.0, -ring_radius, 0.0],
                [0.0, 0.0, ring_radius],
                [0.0, 0.0, -ring_radius],
            ]
            star_points = [r.best_point.tolist()] + [
                (r.best_point + np.array(offset)).tolist() for offset in star_offsets
            ]
            star_lines = [[0, i] for i in range(1, len(star_points))]

            star_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(star_points),
                lines=o3d.utility.Vector2iVector(star_lines),
            )
            star_line_set.colors = o3d.utility.Vector3dVector([color for _ in star_lines])
            geometries.append(star_line_set)

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=best_point_radius * 4)
            axes.translate(r.best_point)
            geometries.append(axes)

        if save_path:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            for geom in geometries:
                vis.add_geometry(geom)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            vis.destroy_window()
            print(f"[CLIPFieldQuery] Saved 3D visualization to {save_path}")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="CLIPField 3D",
            width=1280,
            height=720,
        )

    def _encode_texts(self, texts: List[str]):
        all_clip_queries = self._clip_module.tokenize(texts)
        with torch.no_grad():
            all_clip_tokens = self._clip_model.encode_text(
                all_clip_queries.to(self.device)
            ).float()
            all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
            all_st_tokens = torch.from_numpy(
                self._sentence_model.encode(texts)
            )
            all_st_tokens = F.normalize(all_st_tokens, p=2, dim=-1).to(self.device)
        return all_clip_tokens, all_st_tokens

    def _compute_alignment(self, texts: List[str], visual: bool):
        clip_text_tokens, st_text_tokens = self._encode_texts(texts)

        if visual:
            vision_weight = 10.0
            text_weight = 1.0
        else:
            vision_weight = 1.0
            text_weight = 10.0

        point_opacity = []
        with torch.no_grad():
            for data in tqdm.tqdm(self._points_loader, total=len(self._points_loader)):
                predicted_label_latents, predicted_image_latents = self._label_model(
                    data.to(self.device)
                )
                data_text_tokens = F.normalize(
                    predicted_label_latents, p=2, dim=-1
                ).to(self.device)
                data_visual_tokens = F.normalize(
                    predicted_image_latents, p=2, dim=-1
                ).to(self.device)
                text_alignment = data_text_tokens @ st_text_tokens.T
                visual_alignment = data_visual_tokens @ clip_text_tokens.T
                total_alignment = (
                    text_weight * text_alignment
                ) + (vision_weight * visual_alignment)
                total_alignment /= (text_weight + vision_weight)
                point_opacity.append(total_alignment)

        point_opacity = torch.cat(point_opacity).T
        print(point_opacity.shape)
        return point_opacity

    def _run_queries(
        self, texts: List[str], quantile: float, visual: bool,
    ) -> List[QueryResult]:
        alignment = self._compute_alignment(texts, visual)  # (Q, N)
        all_xyz = self._data._label_xyz.detach().cpu()

        results = []
        for i, text in enumerate(texts):
            q = alignment[i]  # (N,) tensor, on GPU

            alpha = q.detach().cpu().numpy()
            threshold = torch.quantile(q[::10, ...], quantile).cpu().item()

            # a_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min())
            # a_norm_tensor = torch.as_tensor(a_norm)
            # best_idx = int(torch.argmax(a_norm_tensor).item())
            # best_point = all_xyz[best_idx].numpy().copy()

            denom = alpha.max() - alpha.min()
            if denom < 1e-12:
                a_norm = np.zeros_like(alpha)
            else:
                a_norm = (alpha - alpha.min()) / denom
            a_norm_tensor = torch.as_tensor(a_norm)
            topk = min(50, len(a_norm_tensor))
            topk_indices = torch.topk(a_norm_tensor, topk).indices
            topk_scores = a_norm_tensor[topk_indices].numpy()
            topk_points = all_xyz[topk_indices].numpy()
            weights = topk_scores / topk_scores.sum()
            best_point = (topk_points * weights[:, None]).sum(axis=0)

            mask = alpha > threshold
            matched_points = all_xyz[mask].numpy().copy()
            matched_scores = a_norm[mask]

            results.append(QueryResult(
                query=text,
                points=matched_points,
                best_point=best_point,
                scores=matched_scores,
                # best_score=float(a_norm[best_idx]),
                best_score=float(topk_scores.max()),
            ))

        return results


    def visualize(
        self,
        results: Union[QueryResult, List[QueryResult], Dict[str, QueryResult]],
        save_path: Optional[str] = None,
        point_size: float = 0.3,
        highlight_size: float = 8.0,
        best_point_size: float = 80.0,
    ):
        

        if isinstance(results, dict):
            result_list = list(results.values())
        elif isinstance(results, QueryResult):
            result_list = [results]
        else:
            result_list = results

        all_xyz = self._all_xyz.detach().cpu().numpy()
        step = max(1, len(all_xyz) // 50000)  
        scene_pts = all_xyz[::step]

        colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                  "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

        fig = plt.figure(figsize=(14, 6))

        # Coordinate convention (Z-up, confirmed from Open3D view):
        #   X (red)   = depth / forward into scene
        #   Y (green) = right (horizontal)
        #   Z (blue)  = up (sky=high, floor=low)
        # Top-down bird's-eye view  -> X-Y plane (looking down Z)
        # Side elevation view       -> Y-Z plane (looking along X)

        ax1 = fig.add_subplot(121)
        ax1.scatter(scene_pts[:, 0], scene_pts[:, 1],
                    c="#d0d0d0", s=point_size, alpha=0.3)

        for i, r in enumerate(result_list):
            c = colors[i % len(colors)]
            if len(r.points) > 0:
                ax1.scatter(r.points[:, 0], r.points[:, 1],
                            c=c, s=highlight_size, alpha=0.7, label=r.query)
            ax1.scatter(r.best_point[0], r.best_point[1],
                        c=c, s=best_point_size, marker="*", edgecolors="black",
                        linewidths=0.5, zorder=10)
            ax1.annotate(r.query,
                         (r.best_point[0], r.best_point[1]),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=8, fontweight="bold", color=c)

        ax1.set_xlabel("X (depth/forward)")
        ax1.set_ylabel("Y (right)")
        ax1.set_title("Top-down view (X-Y, looking down Z)")
        ax1.set_aspect("equal")
        ax1.legend(fontsize=7, loc="best")

        ax2 = fig.add_subplot(122)
        ax2.scatter(scene_pts[:, 1], scene_pts[:, 2],
                    c="#d0d0d0", s=point_size, alpha=0.3)

        for i, r in enumerate(result_list):
            c = colors[i % len(colors)]
            if len(r.points) > 0:
                ax2.scatter(r.points[:, 1], r.points[:, 2],
                            c=c, s=highlight_size, alpha=0.7, label=r.query)
            ax2.scatter(r.best_point[1], r.best_point[2],
                        c=c, s=best_point_size, marker="*", edgecolors="black",
                        linewidths=0.5, zorder=10)
            ax2.annotate(r.query,
                         (r.best_point[1], r.best_point[2]),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=8, fontweight="bold", color=c)

        ax2.set_xlabel("Y (right)")
        ax2.set_ylabel("Z (up)")
        ax2.set_title("Side elevation view (Y-Z, looking along X)")
        ax2.set_aspect("equal")
        ax2.legend(fontsize=7, loc="best")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[CLIPFieldQuery] Saved visualization to {save_path}")
        plt.show()