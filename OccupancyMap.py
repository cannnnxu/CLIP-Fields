import numpy as np
from typing import Optional, Tuple
from scipy import ndimage


class OccupancyGrid2D:
    def __init__(
        self,
        clip_field,
        resolution: float = 0.05,
        height_filter: float = 0.0,
        inflate_radius: float = 0.0,
        padding: float = 0.5,
    ):
        self.resolution = resolution
        self.height_filter = height_filter
        self.inflate_radius = inflate_radius

        all_xyz = clip_field._all_xyz.detach().cpu().numpy()

        # Filter by height
        mask = all_xyz[:, 1] < height_filter
        filtered = all_xyz[mask]
        pts_xz = filtered[:, [0, 2]]  

        # Grid bounds
        self.origin = pts_xz.min(axis=0) - padding 
        grid_max = pts_xz.max(axis=0) + padding

        self.size = np.ceil((grid_max - self.origin) / resolution).astype(int) 
        self.width = self.size[0]   
        self.height = self.size[1] 

        self._grid = np.zeros((self.height, self.width), dtype=bool)
        indices = np.floor((pts_xz - self.origin) / resolution).astype(int)
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.width) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.height)
        )
        indices = indices[valid]
        self._grid[indices[:, 1], indices[:, 0]] = True

        if inflate_radius > 0:
            inflate_cells = int(np.ceil(inflate_radius / resolution))
            struct = ndimage.generate_binary_structure(2, 2)
            self._grid = ndimage.binary_dilation(
                self._grid, structure=struct, iterations=inflate_cells,
            )

        self._distance_map = ndimage.distance_transform_edt(~self._grid) * resolution

    def _to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world (x, z) to grid indices (col, row)."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((z - self.origin[1]) / self.resolution)
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def is_occupied(self, x: float, z: float) -> bool:
        col, row = self._to_grid(x, z)
        if not self._in_bounds(col, row):
            return True
        return bool(self._grid[row, col])

    def is_free(self, x: float, z: float) -> bool:
        return not self.is_occupied(x, z)

    def distance_to_obstacle(self, x: float, z: float) -> float:
        col, row = self._to_grid(x, z)
        if not self._in_bounds(col, row):
            return 0.0
        return float(self._distance_map[row, col])

    def is_occupied_batch(self, points_xz: np.ndarray) -> np.ndarray:
        cols = ((points_xz[:, 0] - self.origin[0]) / self.resolution).astype(int)
        rows = ((points_xz[:, 1] - self.origin[1]) / self.resolution).astype(int)
        in_bounds = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        result = np.ones(len(points_xz), dtype=bool)  # default: occupied
        result[in_bounds] = self._grid[rows[in_bounds], cols[in_bounds]]
        return result

    def check_path_collision(
        self, start_xz: np.ndarray, end_xz: np.ndarray, steps: int = 50,
    ) -> bool:
        t = np.linspace(0, 1, steps)
        path = start_xz[None, :] * (1 - t[:, None]) + end_xz[None, :] * t[:, None]
        return bool(self.is_occupied_batch(path).any())

    def get_grid(self) -> np.ndarray:
        return self._grid.copy()

    def get_distance_map(self) -> np.ndarray:
        return self._distance_map.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        max_coords = self.origin + self.size * self.resolution
        return self.origin.copy(), max_coords

    def grid_to_world(self, col: int, row: int) -> np.ndarray:
        x = self.origin[0] + (col + 0.5) * self.resolution
        z = self.origin[1] + (row + 0.5) * self.resolution
        return np.array([x, z])

    def get_free_cells(self) -> np.ndarray:
        rows, cols = np.where(~self._grid)
        xs = self.origin[0] + (cols + 0.5) * self.resolution
        zs = self.origin[1] + (rows + 0.5) * self.resolution
        return np.stack([xs, zs], axis=1)
    def visualize(self, goal_xz=None, save_path: Optional[str] = None):
        """Visualize the 2D occupancy grid."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        # Left: binary occupancy
        extent = [
            self.origin[0], self.origin[0] + self.width * self.resolution,
            self.origin[1] + self.height * self.resolution, self.origin[1],
        ]
        ax.imshow(self._grid, cmap="Greys", origin="upper", extent=extent, aspect="equal")
        if goal_xz is not None:
            ax.scatter(goal_xz[0], goal_xz[1], c="green", s=200, marker="*",
                    edgecolors="black", linewidths=0.5, zorder=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"2D occupancy ({self.resolution}m resolution)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[OccupancyGrid2D] Saved to {save_path}")
        plt.show()
