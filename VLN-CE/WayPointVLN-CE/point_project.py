import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt


class DepthToRGBGroundMask:
    def __init__(self, fx, fy, cx, cy, depth_scale=1000.0, camera_height=0, num_boundary_samples=16,
                 save_colored_pcd=True, save_ground_annotated_pcd=True, 
                 save_colored_depth=True, save_sampled_rgb=True):
        """
        初始化相机内参
        
        Args:
            fx, fy: 焦距
            cx, cy: 主点坐标
            depth_scale: 深度图缩放因子 (通常深度值需要除以1000转换为米)
            camera_height: 相机距离地面的高度(米)，用于世界坐标系转换
            num_boundary_samples: 边界采样点数量（默认16个）
            save_colored_pcd: 是否保存彩色点云
            save_ground_annotated_pcd: 是否保存带地面标注的点云
            save_colored_depth: 是否保存彩色深度图
            save_sampled_rgb: 是否保存带采样点的RGB影像
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.camera_height = camera_height
        self.num_boundary_samples = num_boundary_samples
        
        # 数据保存控制参数
        self.save_colored_pcd = save_colored_pcd
        self.save_ground_annotated_pcd = save_ground_annotated_pcd
        self.save_colored_depth = save_colored_depth
        self.save_sampled_rgb = save_sampled_rgb
        
    def depth_to_pointcloud(self, depth_image, rgb_image):
        """
        将深度图转换为彩色3D点云
        
        Args:
            depth_image: 深度图像 (numpy array)
            rgb_image: RGB图像 (numpy array)
            
        Returns:
            points: Nx3 点云数组
            colors: Nx3 颜色数组 (0-1范围)
            valid_u, valid_v: 有效点的像素坐标
        """
        h, w = depth_image.shape[:2]
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        depth = depth_image.flatten()
        
        # 过滤无效深度值
        valid = depth > 0
        u = u[valid]
        v = v[valid]
        depth = depth[valid] / self.depth_scale
        
        # 转换为相机坐标系下的3D坐标
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth
        
        # 转换到世界坐标系（假设相机水平放置）
        # 相机坐标系: x右, y下, z前
        # 世界坐标系: x右, y前, z上
        x_world = x_cam
        y_world = z_cam
        z_world = self.camera_height - y_cam  # 相机高度减去相机y坐标
        
        points = np.stack([x_world, y_world, z_world], axis=1)
        
        # 提取RGB颜色
        rgb_flat = rgb_image.reshape(-1, 3)
        colors = rgb_flat[valid] / 255.0  # 归一化到0-1范围
        # OpenCV使用BGR，转换为RGB
        colors = colors[:, ::-1]
        
        return points, colors, u, v
    
    def detect_ground_plane(self, points, colors, initial_percentile=30, distance_threshold=0.05, 
                           ransac_n=3, num_iterations=1000, normal_z_threshold=0.7):
        """
        使用动态百分位数过滤和RANSAC算法检测地面平面
        
        Args:
            points: Nx3 点云数组 (世界坐标系)
            colors: Nx3 颜色数组
            initial_percentile: 初始百分位数，选择最低的n%点云 (默认10%)
            distance_threshold: RANSAC距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: RANSAC迭代次数
            normal_z_threshold: 法向量z分量阈值，应该接近1表示向上
            
        Returns:
            ground_mask: 布尔数组，标记地面点
            plane_model: 平面方程系数 [a, b, c, d]，满足 ax + by + cz + d = 0
        """
        # 步骤1: 统计所有点的高度
        z_coords = points[:, 2]
        
        # 尝试不同的百分位数，从低到高
        percentiles_to_try = [initial_percentile, 15, 20, 30, 50]
        
        best_plane_model = None
        best_ground_mask = None
        best_normal_z = 0
        
        for percentile in percentiles_to_try:
            # 计算该百分位数对应的高度阈值
            height_threshold = np.percentile(z_coords, percentile)
            
            # 选择低于该阈值的点
            low_points_mask = z_coords < height_threshold
            low_points = points[low_points_mask]
            
            # print(f"\n尝试百分位数 {percentile}%: 高度阈值 = {height_threshold:.3f}m")
            # print(f"  选择了 {len(low_points)} / {len(points)} 个点 ({len(low_points)/len(points)*100:.1f}%)")
            
            if len(low_points) < 10:
                # print(f"  ✗ 点数太少，跳过")
                continue
            
            # 步骤2: 在低点中检测平面
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(low_points)
                
                # 使用RANSAC检测平面
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=ransac_n,
                    num_iterations=num_iterations
                )
                
                # 检查平面法向量
                normal_z = plane_model[2]
                
                # print(f"  平面方程: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
                # print(f"  法向量z分量: {normal_z:.3f} (内点数: {len(inliers)})")
                
                # 如果法向量z分量满足要求（向上）
                if abs(normal_z) >= normal_z_threshold:
                    # print(f"  ✓ 法向量符合要求 (|z| >= {normal_z_threshold})")
                    
                    # 创建完整的地面掩模
                    ground_mask_in_low = np.zeros(len(low_points), dtype=bool)
                    ground_mask_in_low[inliers] = True
                    
                    # 将低点掩模映射回原始点云
                    ground_mask = np.zeros(len(points), dtype=bool)
                    ground_mask[low_points_mask] = ground_mask_in_low
                    
                    # print(f"  检测到地面点数: {np.sum(ground_mask)} / {len(points)} ({np.sum(ground_mask)/len(points)*100:.1f}%)")
                    
                    return ground_mask, plane_model
                else:
                    # print(f"  ✗ 法向量不符合要求 (|z| = {abs(normal_z):.3f} < {normal_z_threshold})")
                    
                    # 记录当前最佳结果
                    if abs(normal_z) > abs(best_normal_z):
                        best_normal_z = normal_z
                        best_plane_model = plane_model
                        
                        ground_mask_in_low = np.zeros(len(low_points), dtype=bool)
                        ground_mask_in_low[inliers] = True
                        best_ground_mask = np.zeros(len(points), dtype=bool)
                        best_ground_mask[low_points_mask] = ground_mask_in_low
                    
            except Exception as e:
                # print(f"  ✗ 平面检测失败: {str(e)}")
                continue
        
        # 如果所有尝试都不满足要求，使用最佳结果
        if best_ground_mask is not None:
            # print(f"\n警告: 未找到完全符合要求的平面，使用最佳结果 (法向量z分量 = {best_normal_z:.3f})")
            # print(f"平面方程: {best_plane_model[0]:.3f}x + {best_plane_model[1]:.3f}y + {best_plane_model[2]:.3f}z + {best_plane_model[3]:.3f} = 0")
            # print(f"检测到地面点数: {np.sum(best_ground_mask)} / {len(points)} ({np.sum(best_ground_mask)/len(points)*100:.1f}%)")
            return best_ground_mask, best_plane_model
        else:
            # 如果完全失败，使用所有点
            # print("\n错误: 所有尝试均失败，使用所有点进行平面检测")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            ground_mask = np.zeros(len(points), dtype=bool)
            ground_mask[inliers] = True
            
            # print(f"平面方程: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
            # print(f"检测到地面点数: {np.sum(ground_mask)} / {len(points)} ({np.sum(ground_mask)/len(points)*100:.1f}%)")
            
            return ground_mask, plane_model
    
    def create_ground_mask_image(self, depth_image, ground_mask, valid_u, valid_v):
        """
        创建地面掩模图像
        
        Args:
            depth_image: 原始深度图
            ground_mask: 地面点布尔掩模
            valid_u, valid_v: 有效点的像素坐标
            
        Returns:
            mask_image: 二值掩模图像 (0或255)
        """
        h, w = depth_image.shape
        mask_image = np.zeros((h, w), dtype=np.uint8)
        
        # 将地面点标记在掩模图像上
        ground_u = valid_u[ground_mask]
        ground_v = valid_v[ground_mask]
        mask_image[ground_v, ground_u] = 255
        
        return mask_image
    
    def sample_ground_boundary_points(self, points, ground_mask, num_samples=5):
        """
        从左向右按角度间隔采样5个路径点
        在地面边缘采样，如果某方向没有地面点则使用固定距离
        
        Args:
            points: Nx3 点云数组 (世界坐标系)
            ground_mask: 地面点布尔掩模
            num_samples: 采样点数量（默认5个）
            
        Returns:
            sampled_points: 5x3 采样点的3D坐标（从左到右）
            sampled_point_ids: 5 采样点的序号 [1,2,3,4,5]
        """
        # 获取地面点
        ground_points = points[ground_mask]
        
        if len(ground_points) == 0:
            # print("警告: 没有地面点，无法采样边界")
            return np.array([]), np.array([])
        
        # 计算地面平均高度
        ground_height = np.mean(ground_points[:, 2])
        ## print(f"地面高度: {ground_height:.3f}m")
        
        # 定义从左到右的5个角度（相对于正前方y轴）
        # 从-40度（左）到+40度（右）
        angles_deg = [-37, -21,-10, 0,10,21, 33]  # 5个角度
        default_distance = 15  # 默认距离15m（如果该方向没有地面点）
        
        sampled_points = []
        sampled_point_ids = [1, 2, 3, 4, 5, 6, 7]

        for i, angle_deg in enumerate(angles_deg):
            point_id = i + 1
            angle_rad = np.radians(angle_deg)
            
            # 在该角度方向上查找地面点
            # 计算所有地面点相对于相机的角度（在x-y平面上）
            ground_points_2d = ground_points[:, :2]  # 只取x, y坐标
            vectors_to_ground = ground_points_2d - np.array([0.0, 0.0])  # 相机位置为原点
            
            # 计算每个地面点的角度和距离（在地面平面上）
            # 角度相对于y轴（正前方），正值向右，负值向左
            ground_angles = np.arctan2(vectors_to_ground[:, 0], vectors_to_ground[:, 1])  # atan2(x, y)
            ground_distances = np.linalg.norm(vectors_to_ground, axis=1)
            
            # 找到与目标角度接近的地面点
            angle_diff = np.abs(ground_angles - angle_rad)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # 处理角度环绕
            
            # 在目标角度±10度范围内的点
            angle_tolerance = np.radians(3)  # 10度
            nearby_mask = angle_diff < angle_tolerance
            
            if np.sum(nearby_mask) > 0:
                # 找到该方向上最远的地面点（边缘点）
                nearby_distances = ground_distances[nearby_mask]
                nearby_ground_points = ground_points[nearby_mask]
                
                farthest_idx = np.argmax(nearby_distances)
                farthest_point = nearby_ground_points[farthest_idx]
                farthest_distance = nearby_distances[farthest_idx]
                
                point_3d = farthest_point
                # print(f"{point_id}号点: 角度{angle_deg:+3d}°, 世界坐标({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}), 距离={farthest_distance:.3f}m")
            else:
                # 该方向没有地面点，使用默认距离
                x = default_distance * np.sin(angle_rad)
                y = default_distance * np.cos(angle_rad)
                z = ground_height
                
                point_3d = np.array([x, y, z])
                # print(f"{point_id}号点: 角度{angle_deg:+3d}°, 世界坐标({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}), 距离={default_distance:.3f}m (默认)")
            
            sampled_points.append(point_3d)
        
        sampled_points = np.array(sampled_points)
        sampled_point_ids = np.array(sampled_point_ids)
        # # print(f"从左到右采样: 成功生成 {len(sampled_points)} 个路径点 (1=左侧, 5=右侧)")
        
        return sampled_points, sampled_point_ids
    
    def project_3d_to_image(self, points_3d):
        """
        将3D点投影到图像坐标
        
        Args:
            points_3d: Nx3 点云数组 (世界坐标系)
            
        Returns:
            image_points: Nx2 图像坐标数组
            valid_mask: 布尔数组，标记有效投影点
        """
        # 世界坐标系转回相机坐标系
        # 世界坐标系: x右, y前, z上
        # 相机坐标系: x右, y下, z前
        x_cam = points_3d[:, 0]
        y_cam = self.camera_height - points_3d[:, 2]  # z_world转为y_cam
        z_cam = points_3d[:, 1]  # y_world转为z_cam
        
        # 过滤掉相机后方的点
        valid_mask = z_cam > 0
        
        # 投影到图像平面
        u = self.fx * x_cam / z_cam + self.cx
        v = self.fy * y_cam / z_cam + self.cy
        
        image_points = np.stack([u, v], axis=1)
        
        return image_points, valid_mask
    
    def visualize_sampled_points(self, rgb_image, sampled_points_3d, sampled_point_ids,
                                 point_radius=12, font_scale=1.0, thickness=3):
        """
        在RGB图像上按点号顺序可视化采样的路径点
        
        Args:
            rgb_image: RGB图像
            sampled_points_3d: Mx3 采样点的3D坐标
            sampled_point_ids: M 采样点的序号
            point_radius: 圆点半径
            font_scale: 字体大小
            thickness: 线条粗细
            
        Returns:
            result_image: 带有可视化的图像
        """
        result_image = rgb_image.copy()
        
        if len(sampled_points_3d) == 0:
            return result_image
        
        # 投影到图像坐标
        image_points, valid_mask = self.project_3d_to_image(sampled_points_3d)
        
        # 按点号顺序绘制每个点
        for i in range(len(sampled_points_3d)):
            if not valid_mask[i]:
                continue
            
            point_id = sampled_point_ids[i]  # 使用实际点号
            center = (int(image_points[i, 0]), int(image_points[i, 1]))
            
            # 检查点是否在图像范围内
            h, w = rgb_image.shape[:2]
            if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
                continue
            
            # 红色实心圆
            cv2.circle(result_image, center, point_radius, (0, 0, 255), -1)
            
            # 绘制序号（红色），位置在圆正上方
            point_text = str(point_id)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            baseline = 0
            text_size = cv2.getTextSize(point_text, font_face, font_scale, thickness)
            text_width = text_size[0][0]
            text_height = text_size[0][1]
            
            text_org = (
                int(center[0] - text_width / 2),
                int(center[1] - point_radius - 4)
            )
            
            cv2.putText(result_image, point_text, text_org, font_face, 
                       font_scale, (0, 0, 255), thickness)
        
        return result_image
    
    def visualize_ground_and_points(self, rgb_image, mask_image, sampled_points_3d, sampled_point_ids,
                                    alpha=0.5, point_radius=10, font_scale=1.0, thickness=3):
        """
        在RGB图像上同时可视化地面掩模和采样点
        
        Args:
            rgb_image: RGB图像
            mask_image: 地面掩模图像
            sampled_points_3d: Mx3 采样点的3D坐标
            sampled_point_ids: M 采样点的序号
            alpha: 地面掩模透明度
            point_radius: 圆点半径
            font_scale: 字体大小
            thickness: 线条粗细
            
        Returns:
            result_image: 合成后的图像
        """
        # 先叠加地面掩模
        result_image = self.project_mask_to_rgb(rgb_image, mask_image, alpha)
        
        if len(sampled_points_3d) == 0:
            return result_image
        
        # 再在上面绘制采样点
        image_points, valid_mask = self.project_3d_to_image(sampled_points_3d)
        
        for i in range(len(sampled_points_3d)):
            if not valid_mask[i]:
                continue
            
            point_id = sampled_point_ids[i]
            center = (int(image_points[i, 0]), int(image_points[i, 1]))
            
            # 检查点是否在图像范围内
            h, w = rgb_image.shape[:2]
            if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
                continue
            
            # 红色实心圆
            cv2.circle(result_image, center, point_radius, (0, 0, 255), -1)
            
            # 绘制序号（红色），位置在圆正上方
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
    
    def create_colored_depth_image(self, depth_image):
        """
        创建彩色深度图
        
        Args:
            depth_image: 深度图像
            
        Returns:
            colored_depth: 彩色深度图
        """
        # 归一化深度图到0-255
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 应用颜色映射（JET色彩映射）
        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return colored_depth
    
    def project_mask_to_rgb(self, rgb_image, mask_image, alpha=0.5):
        """
        将地面掩模投影到RGB图像上进行可视化
        
        Args:
            rgb_image: RGB图像
            mask_image: 二值掩模图像
            alpha: 叠加透明度
            
        Returns:
            result: 叠加后的图像
        """
        result = rgb_image.copy()
        
        # 创建绿色覆盖层
        overlay = result.copy()
        overlay[mask_image > 0] = [0, 255, 0]  # 绿色标记地面
        
        # 混合原图和覆盖层
        result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)
        
        return result
    
    def process_image_pair(self, depth_path, rgb_path, output_dir=None, visualize=True):
        """
        处理深度图和RGB图像对
        
        Args:
            depth_path: 深度图路径
            rgb_path: RGB图像路径
            output_dir: 输出目录
            visualize: 是否可视化结果
        """
        # 读取图像
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        rgb_image = cv2.imread(rgb_path)
        
        if depth_image is None:
            raise ValueError(f"无法读取深度图: {depth_path}")
        if rgb_image is None:
            raise ValueError(f"无法读取RGB图像: {rgb_path}")
        
        # print(f"\n处理图像对: {Path(depth_path).name}")
        # print(f"深度图尺寸: {depth_image.shape}, 类型: {depth_image.dtype}")
        # print(f"RGB图像尺寸: {rgb_image.shape}")
        
        # 转换为彩色点云
        points, colors, valid_u, valid_v = self.depth_to_pointcloud(depth_image, rgb_image)
        # print(f"生成点云: {len(points)} 个点")
        
        # 检测地面
        ground_mask, plane_model = self.detect_ground_plane(points, colors)
        
        # 使用射线投射法在地面边缘采样路径点
        num_samples = getattr(self, 'num_boundary_samples', 16)  # 默认16个
        sampled_points_3d, sampled_point_ids = self.sample_ground_boundary_points(points, ground_mask, num_samples=num_samples)
        
        # 创建地面掩模图像
        mask_image = self.create_ground_mask_image(depth_image, ground_mask, valid_u, valid_v)
        
        # 投影到RGB图像（带地面掩模）
        result_image = self.project_mask_to_rgb(rgb_image, mask_image)
        
        # 在原始RGB图像上按点号顺序可视化采样点（不带地面掩模）
        result_image_with_points = self.visualize_sampled_points(rgb_image, sampled_points_3d, sampled_point_ids)
        
        # 创建带地面掩模和采样点的合成图像
        result_image_with_ground_and_points = self.visualize_ground_and_points(
            rgb_image, mask_image, sampled_points_3d, sampled_point_ids
        )
        
        # 创建彩色深度图
        colored_depth = self.create_colored_depth_image(depth_image)
        
        # 根据配置保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            base_name = Path(depth_path).stem
            
            # 基础文件（总是保存）
            mask_path = output_dir / f"{base_name}_ground_mask.png"
            result_path = output_dir / f"{base_name}_result.png"
            cv2.imwrite(str(mask_path), mask_image)
            cv2.imwrite(str(result_path), result_image)
            # print(f"已保存掩模: {mask_path}")
            # print(f"已保存结果: {result_path}")
            
            # 可选文件
            if self.save_sampled_rgb:
                result_points_path = output_dir / f"{base_name}_result_with_points.png"
                cv2.imwrite(str(result_points_path), result_image_with_points)
                # print(f"已保存带采样点的RGB影像: {result_points_path}")
                
                # 保存带地面掩模和采样点的合成图像
                result_ground_points_path = output_dir / f"{base_name}_result_ground_with_points.png"
                cv2.imwrite(str(result_ground_points_path), result_image_with_ground_and_points)
                # print(f"已保存地面+采样点合成图像: {result_ground_points_path}")
            
            if self.save_colored_depth:
                colored_depth_path = output_dir / f"{base_name}_colored_depth.png"
                cv2.imwrite(str(colored_depth_path), colored_depth)
                # print(f"已保存彩色深度图: {colored_depth_path}")
        
        # 可视化
        if visualize:
            self.visualize_results(rgb_image, depth_image, mask_image, result_image_with_points, 
                                  points, colors, ground_mask, sampled_points_3d, output_dir)
        
        return mask_image, result_image_with_points, points, colors, ground_mask, sampled_points_3d, sampled_point_ids
    
    def visualize_results(self, rgb_image, depth_image, mask_image, result_image, 
                         points, colors, ground_mask, sampled_points_3d, output_dir=None):
        """
        可视化处理结果（保存为图片而不是显示）
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 原始RGB图像
        axes[0, 0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始RGB图像')
        axes[0, 0].axis('off')
        
        # 深度图
        axes[0, 1].imshow(depth_image, cmap='jet')
        axes[0, 1].set_title('深度图')
        axes[0, 1].axis('off')
        
        # 地面掩模
        axes[1, 0].imshow(mask_image, cmap='gray')
        axes[1, 0].set_title('地面掩模')
        axes[1, 0].axis('off')
        
        # 结果图像（带采样点）
        axes[1, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('地面检测结果 (绿色=地面, 红点=路径点)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化图像
        if output_dir:
            viz_path = Path(output_dir) / "visualization_summary.png"
            plt.savefig(str(viz_path), dpi=100, bbox_inches='tight')
            # print(f"已保存可视化图像: {viz_path}")
        
        plt.close(fig)
        
        # 保存3D彩色点云（不显示GUI）
        self.save_pointcloud(points, colors, ground_mask, sampled_points_3d, output_dir)
    
    def save_pointcloud(self, points, colors, ground_mask, sampled_points_3d, output_dir=None):
        """
        根据配置保存3D彩色点云为PLY文件（不显示GUI）
        """
        if not output_dir:
            return
        
        # 保存原始彩色点云（如果启用）
        if self.save_colored_pcd:
            pcd_color = o3d.geometry.PointCloud()
            pcd_color.points = o3d.utility.Vector3dVector(points)
            pcd_color.colors = o3d.utility.Vector3dVector(colors)
            
            ply_color_path = Path(output_dir) / "pointcloud_colored.ply"
            o3d.io.write_point_cloud(str(ply_color_path), pcd_color)
            # print(f"已保存彩色点云文件: {ply_color_path}")
        
        # 保存带地面标注的点云（如果启用）
        if self.save_ground_annotated_pcd:
            pcd_ground = o3d.geometry.PointCloud()
            pcd_ground.points = o3d.utility.Vector3dVector(points)
            
            colors_with_ground = colors.copy()
            colors_with_ground[ground_mask] = [0, 1, 0]  # 地面标记为绿色
            pcd_ground.colors = o3d.utility.Vector3dVector(colors_with_ground)
            
            ply_ground_path = Path(output_dir) / "pointcloud_with_ground.ply"
            o3d.io.write_point_cloud(str(ply_ground_path), pcd_ground)
            # print(f"已保存地面标注点云文件: {ply_ground_path} (绿色=地面)")
        
        # 保存单独的地面点云（只包含地面点）
        if np.sum(ground_mask) > 0:
            ground_points = points[ground_mask]
            ground_colors = colors[ground_mask]
            
            pcd_ground_only = o3d.geometry.PointCloud()
            pcd_ground_only.points = o3d.utility.Vector3dVector(ground_points)
            pcd_ground_only.colors = o3d.utility.Vector3dVector(ground_colors)
            
            ply_ground_only_path = Path(output_dir) / "pointcloud_ground_only.ply"
            o3d.io.write_point_cloud(str(ply_ground_only_path), pcd_ground_only)
            # print(f"已保存地面点云文件: {ply_ground_only_path} (仅地面点: {len(ground_points)} 个)")
    
    def batch_process(self, depth_dir, rgb_dir, output_dir, visualize_first=True):
        """
        批量处理所有图像对
        
        Args:
            depth_dir: 深度图目录
            rgb_dir: RGB图像目录
            output_dir: 输出目录
            visualize_first: 是否只可视化第一对图像
        """
        depth_dir = Path(depth_dir)
        rgb_dir = Path(rgb_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 获取所有深度图
        depth_files = sorted(list(depth_dir.glob('*.jpg')) + list(depth_dir.glob('*.png')))
        
        # print(f"找到 {len(depth_files)} 个深度图")
        
        for i, depth_path in enumerate(depth_files):
            # 找到对应的RGB图像
            rgb_path = rgb_dir / depth_path.name
            
            if not rgb_path.exists():
                # print(f"警告: 未找到对应的RGB图像 {rgb_path}")
                continue
            
            # 处理图像对
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

def process_depth_rgb_simple(depth_image, rgb_image, 
                             fx=320.0, fy=320.0, cx=319.5, cy=239.5,
                             depth_scale=10.0, camera_height=2.4,
                             num_samples=7):
    """
    简单易用的函数：传入深度图和RGB图像，返回带采样点的图像
    
    Args:
        depth_image: 深度图 (numpy array, H×W, uint16)
        rgb_image: RGB图像 (numpy array, H×W×3, uint8, BGR格式)
        fx, fy: 焦距 (默认320.0)
        cx, cy: 主点坐标 (默认319.5, 239.5)
        depth_scale: 深度缩放因子 (默认10.0)
        camera_height: 相机高度，单位米 (默认2.4)
        num_samples: 采样点数量 (默认7个)
        
    Returns:
        result_image: 带采样点标注的RGB图像 (numpy array, H×W×3, uint8, BGR格式)
        sampled_points_3d: 采样点的3D世界坐标 (numpy array, N×3)
        sampled_point_ids: 采样点的序号 (numpy array, N)
        
    Example:
        >>> import cv2
        >>> depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
        >>> rgb = cv2.imread('rgb.jpg')
        >>> result, points_3d, point_ids = process_depth_rgb_simple(depth, rgb)
        >>> cv2.imwrite('output.jpg', result)
    """
    # 创建处理器
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
    
    # 转换为点云
    points, colors, valid_u, valid_v = processor.depth_to_pointcloud(depth_image, rgb_image)
    
    # 检测地面
    ground_mask, plane_model = processor.detect_ground_plane(points, colors)
    
    # 采样路径点
    sampled_points_3d, sampled_point_ids = processor.sample_ground_boundary_points(
        points, ground_mask, num_samples=num_samples
    )
    
    # 可视化采样点
    result_image = processor.visualize_sampled_points(
        rgb_image, sampled_points_3d, sampled_point_ids
    )
    
    return result_image, sampled_points_3d, sampled_point_ids


def process_depth_rgb_with_ground(depth_image, rgb_image, 
                                   fx=320.0, fy=320.0, cx=319.5, cy=239.5,
                                   depth_scale=10.0, camera_height=2.4,
                                   num_samples=7, alpha=0.5):
    """
    传入深度图和RGB图像，返回带地面掩模和采样点的图像
    
    Args:
        depth_image: 深度图 (numpy array, H×W, uint16)
        rgb_image: RGB图像 (numpy array, H×W×3, uint8, BGR格式)
        fx, fy: 焦距 (默认320.0)
        cx, cy: 主点坐标 (默认319.5, 239.5)
        depth_scale: 深度缩放因子 (默认10.0)
        camera_height: 相机高度，单位米 (默认2.4)
        num_samples: 采样点数量 (默认7个)
        alpha: 地面掩模透明度 (默认0.5)
        
    Returns:
        result_image: 带地面掩模和采样点的图像 (numpy array, H×W×3, uint8, BGR格式)
        sampled_points_3d: 采样点的3D世界坐标 (numpy array, N×3)
        sampled_point_ids: 采样点的序号 (numpy array, N)
        ground_mask_image: 地面掩模图像 (numpy array, H×W, uint8)
        
    Example:
        >>> import cv2
        >>> depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
        >>> rgb = cv2.imread('rgb.jpg')
        >>> result, points_3d, point_ids, mask = process_depth_rgb_with_ground(depth, rgb)
        >>> cv2.imwrite('output_with_ground.jpg', result)
    """
    # 创建处理器
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
    
    # 转换为点云
    points, colors, valid_u, valid_v = processor.depth_to_pointcloud(depth_image, rgb_image)
    
    # 检测地面
    ground_mask, plane_model = processor.detect_ground_plane(points, colors)
    
    # 采样路径点
    sampled_points_3d, sampled_point_ids = processor.sample_ground_boundary_points(
        points, ground_mask, num_samples=num_samples
    )
    
    # 创建地面掩模图像
    mask_image = processor.create_ground_mask_image(depth_image, ground_mask, valid_u, valid_v)
    
    # 可视化地面和采样点
    result_image = processor.visualize_ground_and_points(
        rgb_image, mask_image, sampled_points_3d, sampled_point_ids, alpha=alpha
    )
    
    return result_image, sampled_points_3d, sampled_point_ids, mask_image