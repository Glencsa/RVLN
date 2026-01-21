import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class DepthToRGBGroundMask:
    """
    Utility for converting depth and RGB images to 3D point clouds, detecting ground planes,
    sampling boundary points, and visualizing results.
    """
    def __init__(self, fx: float, fy: float, cx: float, cy: float, depth_scale: float = 1000.0, camera_height: float = 0.0, num_boundary_samples: int = 16,
                 save_colored_pcd: bool = True, save_ground_annotated_pcd: bool = True, 
                 save_colored_depth: bool = True, save_sampled_rgb: bool = True):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.camera_height = camera_height
        self.num_boundary_samples = num_boundary_samples
        self.save_colored_pcd = save_colored_pcd
        self.save_ground_annotated_pcd = save_ground_annotated_pcd
        self.save_colored_depth = save_colored_depth
        self.save_sampled_rgb = save_sampled_rgb

    def depth_to_pointcloud(self, depth_image: np.ndarray, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert depth and RGB images to a colored 3D point cloud.
        Returns points (Nx3), colors (Nx3), and valid pixel coordinates (u, v).
        """
        h, w = depth_image.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        depth = depth_image.flatten()
        valid = depth > 0
        u = u[valid]
        v = v[valid]
        depth = depth[valid] / self.depth_scale
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth
        x_world = x_cam
        y_world = z_cam
        z_world = self.camera_height - y_cam
        points = np.stack([x_world, y_world, z_world], axis=1)
        rgb_flat = rgb_image.reshape(-1, 3)
        colors = rgb_flat[valid] / 255.0
        colors = colors[:, ::-1]
        return points, colors, u, v

    def detect_ground_plane(self, points: np.ndarray, colors: np.ndarray, initial_percentile: int = 30, distance_threshold: float = 0.05, 
                           ransac_n: int = 3, num_iterations: int = 1000, normal_z_threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the ground plane using percentile filtering and RANSAC.
        Returns a boolean mask for ground points and the plane model coefficients.
        """
        z_coords = points[:, 2]
        percentiles_to_try = [initial_percentile, 15, 20, 30, 50]
        best_plane_model = None
        best_ground_mask = None
        best_normal_z = 0
        for percentile in percentiles_to_try:
            height_threshold = np.percentile(z_coords, percentile)
            low_points_mask = z_coords < height_threshold
            low_points = points[low_points_mask]
            if len(low_points) < 10:
                continue
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(low_points)
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations
                )
                normal_z = plane_model[2]
                if abs(normal_z) >= normal_z_threshold:
                    ground_mask_in_low = np.zeros(len(low_points), dtype=bool)
                    ground_mask_in_low[inliers] = True
                    ground_mask = np.zeros(len(points), dtype=bool)
                    ground_mask[low_points_mask] = ground_mask_in_low
                    return ground_mask, plane_model
                else:
                    if abs(normal_z) > abs(best_normal_z):
                        best_normal_z = normal_z
                        best_plane_model = plane_model
                        ground_mask_in_low = np.zeros(len(low_points), dtype=bool)
                        ground_mask_in_low[inliers] = True
                        best_ground_mask = np.zeros(len(points), dtype=bool)
                        best_ground_mask[low_points_mask] = ground_mask_in_low
            except Exception:
                continue
        if best_ground_mask is not None:
            return best_ground_mask, best_plane_model
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            ground_mask = np.zeros(len(points), dtype=bool)
            ground_mask[inliers] = True
            return ground_mask, plane_model

    def create_ground_mask_image(self, depth_image: np.ndarray, ground_mask: np.ndarray, valid_u: np.ndarray, valid_v: np.ndarray) -> np.ndarray:
        """
        Create a binary mask image for ground points.
        """
        h, w = depth_image.shape
        mask_image = np.zeros((h, w), dtype=np.uint8)
        ground_u = valid_u[ground_mask]
        ground_v = valid_v[ground_mask]
        mask_image[ground_v, ground_u] = 255
        return mask_image

    def sample_ground_boundary_points(self, points: np.ndarray, ground_mask: np.ndarray, num_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample boundary points along the ground from left to right at fixed angles.
        Returns sampled 3D points and their IDs.
        """
        ground_points = points[ground_mask]
        if len(ground_points) == 0:
            return np.array([]), np.array([])
        ground_height = np.mean(ground_points[:, 2])
        angles_deg = [-37, -21, -10, 0, 10, 21, 33]
        default_distance = 15
        sampled_points = []
        sampled_point_ids = [1, 2, 3, 4, 5, 6, 7]
        for i, angle_deg in enumerate(angles_deg):
            point_id = i + 1
            angle_rad = np.radians(angle_deg)
            ground_points_2d = ground_points[:, :2]
            vectors_to_ground = ground_points_2d - np.array([0.0, 0.0])
            ground_angles = np.arctan2(vectors_to_ground[:, 0], vectors_to_ground[:, 1])
            ground_distances = np.linalg.norm(vectors_to_ground, axis=1)
            angle_diff = np.abs(ground_angles - angle_rad)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            angle_tolerance = np.radians(3)
            nearby_mask = angle_diff < angle_tolerance
            if np.sum(nearby_mask) > 0:
                nearby_distances = ground_distances[nearby_mask]
                nearby_ground_points = ground_points[nearby_mask]
                farthest_idx = np.argmax(nearby_distances)
                point_3d = nearby_ground_points[farthest_idx]
            else:
                x = default_distance * np.sin(angle_rad)
                y = default_distance * np.cos(angle_rad)
                z = ground_height
                point_3d = np.array([x, y, z])
            sampled_points.append(point_3d)
        sampled_points = np.array(sampled_points)
        sampled_point_ids = np.array(sampled_point_ids)
        return sampled_points, sampled_point_ids

    def project_3d_to_image(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to image coordinates.
        Returns image points and a boolean mask for valid projections.
        """
        x_cam = points_3d[:, 0]
        y_cam = self.camera_height - points_3d[:, 2]
        z_cam = points_3d[:, 1]
        valid_mask = z_cam > 0
        u = self.fx * x_cam / z_cam + self.cx
        v = self.fy * y_cam / z_cam + self.cy
        image_points = np.stack([u, v], axis=1)
        return image_points, valid_mask

    def visualize_sampled_points(self, rgb_image: np.ndarray, sampled_points_3d: np.ndarray, sampled_point_ids: np.ndarray,
                                 point_radius: int = 12, font_scale: float = 1.0, thickness: int = 3) -> np.ndarray:
        """
        Visualize sampled points on the RGB image.
        """
        result_image = rgb_image.copy()
        if len(sampled_points_3d) == 0:
            return result_image
        image_points, valid_mask = self.project_3d_to_image(sampled_points_3d)
        for i in range(len(sampled_points_3d)):
            if not valid_mask[i]:
                continue
            point_id = sampled_point_ids[i]
            center = (int(image_points[i, 0]), int(image_points[i, 1]))
            h, w = rgb_image.shape[:2]
            if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
                continue
            cv2.circle(result_image, center, point_radius, (0, 0, 255), -1)
            point_text = str(point_id)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(point_text, font_face, font_scale, thickness)
            text_width = text_size[0][0]
            text_org = (
                int(center[0] - text_width / 2),
                int(center[1] - point_radius - 4)
            )
            cv2.putText(result_image, point_text, text_org, font_face, 
                       font_scale, (0, 0, 255), thickness)
        return result_image

    def visualize_ground_and_points(self, rgb_image: np.ndarray, mask_image: np.ndarray, sampled_points_3d: np.ndarray, sampled_point_ids: np.ndarray,
                                    alpha: float = 0.5, point_radius: int = 10, font_scale: float = 1.0, thickness: int = 3) -> np.ndarray:
        """
        Visualize both ground mask and sampled points on the RGB image.
        """
        result_image = self.project_mask_to_rgb(rgb_image, mask_image, alpha)
        if len(sampled_points_3d) == 0:
            return result_image
        image_points, valid_mask = self.project_3d_to_image(sampled_points_3d)
        for i in range(len(sampled_points_3d)):
            if not valid_mask[i]:
                continue
            point_id = sampled_point_ids[i]
            center = (int(image_points[i, 0]), int(image_points[i, 1]))
            h, w = rgb_image.shape[:2]
            if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
                continue
            cv2.circle(result_image, center, point_radius, (0, 0, 255), -1)
            point_text = str(point_id)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(point_text, font_face, font_scale, thickness)
            text_width = text_size[0][0]
            text_org = (
                int(center[0] - text_width / 2),
                int(center[1] - point_radius - 4)
            )
            cv2.putText(result_image, point_text, text_org, font_face, 
                       font_scale, (0, 0, 255), thickness)
        return result_image

    def create_colored_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Create a color-mapped depth image for visualization.
        """
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return colored_depth

    def project_mask_to_rgb(self, rgb_image: np.ndarray, mask_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay a binary mask on the RGB image with transparency.
        """
        result = rgb_image.copy()
        overlay = result.copy()
        overlay[mask_image > 0] = [0, 255, 0]
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        return result

    def process_image_pair(self, depth_path: str, rgb_path: str, output_dir: Optional[str] = None, visualize: bool = True):
        """
        Process a pair of depth and RGB images, save and/or visualize results.
        """
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        rgb_image = cv2.imread(rgb_path)
        if depth_image is None:
            raise ValueError(f"Cannot read depth image: {depth_path}")
        if rgb_image is None:
            raise ValueError(f"Cannot read RGB image: {rgb_path}")
        points, colors, valid_u, valid_v = self.depth_to_pointcloud(depth_image, rgb_image)
        ground_mask, plane_model = self.detect_ground_plane(points, colors)
        num_samples = getattr(self, 'num_boundary_samples', 16)
        sampled_points_3d, sampled_point_ids = self.sample_ground_boundary_points(points, ground_mask, num_samples=num_samples)
        mask_image = self.create_ground_mask_image(depth_image, ground_mask, valid_u, valid_v)
        result_image = self.project_mask_to_rgb(rgb_image, mask_image)
        result_image_with_points = self.visualize_sampled_points(rgb_image, sampled_points_3d, sampled_point_ids)
        result_image_with_ground_and_points = self.visualize_ground_and_points(
            rgb_image, mask_image, sampled_points_3d, sampled_point_ids
        )
        colored_depth = self.create_colored_depth_image(depth_image)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            base_name = Path(depth_path).stem
            mask_path = output_dir / f"{base_name}_ground_mask.png"
            result_path = output_dir / f"{base_name}_result.png"
            cv2.imwrite(str(mask_path), mask_image)
            cv2.imwrite(str(result_path), result_image)
            if self.save_sampled_rgb:
                result_points_path = output_dir / f"{base_name}_result_with_points.png"
                cv2.imwrite(str(result_points_path), result_image_with_points)
                result_ground_points_path = output_dir / f"{base_name}_result_ground_with_points.png"
                cv2.imwrite(str(result_ground_points_path), result_image_with_ground_and_points)
            if self.save_colored_depth:
                colored_depth_path = output_dir / f"{base_name}_colored_depth.png"
                cv2.imwrite(str(colored_depth_path), colored_depth)
        if visualize:
            self.visualize_results(rgb_image, depth_image, mask_image, result_image_with_points, 
                                  points, colors, ground_mask, sampled_points_3d, output_dir)
        return mask_image, result_image_with_points, points, colors, ground_mask, sampled_points_3d, sampled_point_ids

    def visualize_results(self, rgb_image: np.ndarray, depth_image: np.ndarray, mask_image: np.ndarray, result_image: np.ndarray, 
                         points: np.ndarray, colors: np.ndarray, ground_mask: np.ndarray, sampled_points_3d: np.ndarray, output_dir: Optional[str] = None):
        """
        Save a summary visualization of the results as an image file.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(depth_image, cmap='jet')
        axes[0, 1].set_title('Depth Image')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(mask_image, cmap='gray')
        axes[1, 0].set_title('Ground Mask')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Ground Detection (green=ground, red=points)')
        axes[1, 1].axis('off')
        plt.tight_layout()
        if output_dir:
            viz_path = Path(output_dir) / "visualization_summary.png"
            plt.savefig(str(viz_path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        self.save_pointcloud(points, colors, ground_mask, sampled_points_3d, output_dir)

    def save_pointcloud(self, points: np.ndarray, colors: np.ndarray, ground_mask: np.ndarray, sampled_points_3d: np.ndarray, output_dir: Optional[str] = None):
        """
        Save colored point clouds as PLY files if enabled.
        """
        if not output_dir:
            return
        if self.save_colored_pcd:
            pcd_color = o3d.geometry.PointCloud()
            pcd_color.points = o3d.utility.Vector3dVector(points)
            pcd_color.colors = o3d.utility.Vector3dVector(colors)
            ply_color_path = Path(output_dir) / "pointcloud_colored.ply"
            o3d.io.write_point_cloud(str(ply_color_path), pcd_color)
        if self.save_ground_annotated_pcd:
            pcd_ground = o3d.geometry.PointCloud()
            pcd_ground.points = o3d.utility.Vector3dVector(points)
            colors_with_ground = colors.copy()
            colors_with_ground[ground_mask] = [0, 1, 0]
            pcd_ground.colors = o3d.utility.Vector3dVector(colors_with_ground)
            ply_ground_path = Path(output_dir) / "pointcloud_with_ground.ply"
            o3d.io.write_point_cloud(str(ply_ground_path), pcd_ground)
        if np.sum(ground_mask) > 0:
            ground_points = points[ground_mask]
            ground_colors = colors[ground_mask]
            pcd_ground_only = o3d.geometry.PointCloud()
            pcd_ground_only.points = o3d.utility.Vector3dVector(ground_points)
            pcd_ground_only.colors = o3d.utility.Vector3dVector(ground_colors)
            ply_ground_only_path = Path(output_dir) / "pointcloud_ground_only.ply"
            o3d.io.write_point_cloud(str(ply_ground_only_path), pcd_ground_only)

    def batch_process(self, depth_dir: str, rgb_dir: str, output_dir: str, visualize_first: bool = True):
        """
        Batch process all image pairs in the given directories.
        """
        depth_dir = Path(depth_dir)
        rgb_dir = Path(rgb_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        depth_files = sorted(list(depth_dir.glob('*.jpg')) + list(depth_dir.glob('*.png')))
        for i, depth_path in enumerate(depth_files):
            rgb_path = rgb_dir / depth_path.name
            if not rgb_path.exists():
                continue
            visualize = (i == 0 and visualize_first)
            self.process_image_pair(
                str(depth_path), 
                str(rgb_path), 
                output_dir=output_dir,
                visualize=visualize
            )


def main():
    """
    主函数 - 示例用法
    """
    # ========== 配置相机内参 ==========
    # 请根据你的相机实际参数修改这些值
    # 如果不知道准确值，可以使用图像中心作为主点，焦距设为图像宽度
    fx = 320.0  # 焦距 x
    fy = 320.0  # 焦距 y
    cx = 319.5  # 主点 x (通常是图像宽度的一半)
    cy = 239.5  # 主点 y (通常是图像高度的一半)
    depth_scale = 10.0  # 深度缩放因子 (深度值除以此值得到米)
    camera_height = 2.4  # 相机距离地面高度(米)
    
    # ========== 配置采样参数 ==========
    num_boundary_samples = 32  # 地面边界采样点数量（可修改：8, 12, 16, 24, 32等）
    
    # ========== 配置数据保存选项 ==========
    save_colored_pcd = True            # 是否保存彩色点云
    save_ground_annotated_pcd = True    # 是否保存带地面标注的点云
    save_colored_depth = False           # 是否保存彩色深度图
    save_sampled_rgb = True            # 是否保存带采样点的RGB影像
    
    # ========== 配置路径 ==========
    depth_dir = "depth"  # 深度图目录
    rgb_dir = "rgb"      # RGB图像目录
    output_dir = "output"  # 输出目录
    
    # ========== 创建处理器 ==========
    processor = DepthToRGBGroundMask(
        fx, fy, cx, cy, depth_scale, camera_height, num_boundary_samples,
        save_colored_pcd=save_colored_pcd,
        save_ground_annotated_pcd=save_ground_annotated_pcd,
        save_colored_depth=save_colored_depth,
        save_sampled_rgb=save_sampled_rgb
    )
    
    # ========== 批量处理 ==========
    # 设置为True将为第一张图片生成可视化总结图和点云文件
    processor.batch_process(depth_dir, rgb_dir, output_dir, visualize_first=True)
    
    # print("\n处理完成!")
    # print("\n输出文件说明:")
    # print("基础文件（总是保存）:")
    # print("  - *_ground_mask.png: 地面二值掩模")
    # print("  - *_result.png: RGB图像上的地面标注(绿色)")
    # print("  - visualization_summary.png: 第一张图的可视化总结")
    
    # print("\n可选文件（根据配置）:")
    # if save_sampled_rgb:
        # print("  - *_result_with_points.png: 带边界采样点的RGB影像(红点+序号)")
        # print("  - *_result_ground_with_points.png: 地面掩模+采样点合成图像(绿色地面+红点)")
    # if save_colored_depth:
        # print("  - *_colored_depth.png: 彩色深度图")
    # if save_colored_pcd:
        # print("  - pointcloud_colored.ply: 原始彩色3D点云")
    # if save_ground_annotated_pcd:
        # print("  - pointcloud_with_ground.ply: 带地面标注的彩色3D点云(绿色=地面)")
        # print("  - pointcloud_ground_only.ply: 单独的地面点云(仅包含地面点)")
    
    # print("\n算法说明:")
    # print("  - 使用射线投射法从相机位置向前方视野按角度间隔发射射线")
    # print("  - 当射线碰到地面外边缘时记录为采样点")
    # print("  - 采样点按点号顺序标注在RGB图像上")
    # print("\n提示: 可以使用 CloudCompare, MeshLab 等软件打开PLY文件查看3D点云")


if __name__ == "__main__":
    main()


# ==================== 简单易用的接口函数 ====================

def process_depth_rgb_simple(depth_image: np.ndarray, rgb_image: np.ndarray, 
                             fx: float = 320.0, fy: float = 320.0, cx: float = 319.5, cy: float = 239.5,
                             depth_scale: float = 10.0, camera_height: float = 2.4,
                             num_samples: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple interface: process depth and RGB images, return RGB with sampled points, 3D points, and IDs.
    """
    processor = DepthToRGBGroundMask(
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=depth_scale,
        camera_height=camera_height,
        num_boundary_samples=num_samples,
        save_colored_pcd=False,
        save_ground_annotated_pcd=False,
        save_colored_depth=False,
        save_sampled_rgb=False
    )
    points, colors, valid_u, valid_v = processor.depth_to_pointcloud(depth_image, rgb_image)
    ground_mask, plane_model = processor.detect_ground_plane(points, colors)
    sampled_points_3d, sampled_point_ids = processor.sample_ground_boundary_points(
        points, ground_mask, num_samples=num_samples
    )
    result_image = processor.visualize_sampled_points(
        rgb_image, sampled_points_3d, sampled_point_ids
    )
    return result_image, sampled_points_3d, sampled_point_ids

def process_depth_rgb_highlight(depth_image: np.ndarray, rgb_image: np.ndarray, selected_ids: List[int],
                            fx: float = 320.0, fy: float = 320.0, cx: float = 319.5, cy: float = 239.5,
                            depth_scale: float = 10.0, camera_height: float = 2.4,
                            num_samples: int = 7,
                            normal_radius: int = 11, highlight_radius: int = 11,
                            normal_color: Tuple[int, int, int] = (255, 0, 0), highlight_color: Tuple[int, int, int] = (0, 255, 0),
                            font_scale: float = 1.0, thickness: int = 3) -> np.ndarray:
    """
    Visualize sampled points on RGB image, highlighting selected IDs.
    """
    sel_set = set(int(x) for x in (selected_ids or []))
    processor = DepthToRGBGroundMask(
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=depth_scale,
        camera_height=camera_height,
        num_boundary_samples=num_samples,
        save_colored_pcd=False,
        save_ground_annotated_pcd=False,
        save_colored_depth=False,
        save_sampled_rgb=False
    )
    points, colors, valid_u, valid_v = processor.depth_to_pointcloud(depth_image, rgb_image)
    ground_mask, plane_model = processor.detect_ground_plane(points, colors)
    sampled_points_3d, sampled_point_ids = processor.sample_ground_boundary_points(points, ground_mask, num_samples=num_samples)
    result_image = rgb_image.copy()
    if len(sampled_points_3d) == 0:
        return result_image
    image_points, valid_mask = processor.project_3d_to_image(sampled_points_3d)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(sampled_points_3d)):
        if not valid_mask[i]:
            continue
        point_id = int(sampled_point_ids[i])
        center = (int(round(image_points[i, 0])), int(round(image_points[i, 1])))
        h, w = rgb_image.shape[:2]
        if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
            continue
        if point_id in sel_set:
            color = highlight_color
            radius = highlight_radius
            text_thickness = max(2, thickness)
        else:
            color = normal_color
            radius = normal_radius
            text_thickness = thickness
        cv2.circle(result_image, center, radius, color, -1)
        point_text = str(point_id)
        text_size = cv2.getTextSize(point_text, font_face, font_scale, text_thickness)[0]
        text_org = (int(center[0] - text_size[0] / 2), int(center[1] - radius - 4))
        cv2.putText(result_image, point_text, text_org, font_face, font_scale, color, text_thickness, cv2.LINE_AA)
    return result_image

def process_depth_rgb_with_ground(depth_image: np.ndarray, rgb_image: np.ndarray, 
                                   fx: float = 320.0, fy: float = 320.0, cx: float = 319.5, cy: float = 239.5,
                                   depth_scale: float = 10.0, camera_height: float = 2.4,
                                   num_samples: int = 7, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process depth and RGB images, return RGB with ground mask and sampled points, 3D points, IDs, and ground mask image.
    """
    processor = DepthToRGBGroundMask(
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=depth_scale,
        camera_height=camera_height,
        num_boundary_samples=num_samples,
        save_colored_pcd=False,
        save_ground_annotated_pcd=False,
        save_colored_depth=False,
        save_sampled_rgb=False
    )
    points, colors, valid_u, valid_v = processor.depth_to_pointcloud(depth_image, rgb_image)
    ground_mask, plane_model = processor.detect_ground_plane(points, colors)
    sampled_points_3d, sampled_point_ids = processor.sample_ground_boundary_points(
        points, ground_mask, num_samples=num_samples
    )
    mask_image = processor.create_ground_mask_image(depth_image, ground_mask, valid_u, valid_v)
    result_image = processor.visualize_ground_and_points(
        rgb_image, mask_image, sampled_points_3d, sampled_point_ids, alpha=alpha
    )
    return result_image, sampled_points_3d, sampled_point_ids, mask_image