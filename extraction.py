# import os
# import cv2
# import torch
# import numpy as np
# import clip
# from PIL import Image
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")

# # ===================== 精细化配置（针对512维特征 + 5秒窗口） =====================
# class CLIPVideoFeatureConfig:
#     """针对512维CLIP特征 + 5秒时间窗口的配置类"""
#     def __init__(self):
#         # CLIP模型配置（固定ViT-B/32以保证512维）
#         self.clip_model_name = "ViT-B/32"
#         self.feature_dim = 512  # 固定512维
        
#         # 时间窗口配置（核心修改：5秒窗口）
#         self.window_seconds = 5  # 每个窗口5秒
#         self.sample_rate_per_window = 2  # 每个窗口内的采样率（每隔2帧取1帧）
#         self.max_frames_per_window = 50  # 每个窗口最多提取50帧
        
#         # 预处理配置（严格匹配CLIP的要求）
#         self.clip_image_size = 224  # ViT-B/32要求的输入尺寸
#         self.clip_mean = (0.48145466, 0.4578275, 0.40821073)  # CLIP官方均值
#         self.clip_std = (0.26862954, 0.26130258, 0.27577711)   # CLIP官方标准差
        
#         # 保存配置
#         self.save_dtype = np.float32  # 512维特征用float32足够，节省空间
#         self.overwrite = False        # 不覆盖已存在的npy文件

# # ===================== 按5秒窗口提取帧 =====================
# def split_video_into_5s_windows(video_path, config):
#     """
#     将视频切分为5秒窗口，并提取每个窗口的帧
#     :return: 字典 {窗口编号: (帧列表, 窗口起始时间, 窗口结束时间)}
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"视频文件无法打开: {video_path}")
    
#     # 获取视频基础信息
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     duration = total_frames / fps if fps > 0 else 0
#     print(f"视频基础信息：总帧数={total_frames}, FPS={fps:.2f}, 总时长={duration:.2f}秒")
    
#     # 计算5秒窗口对应的帧数
#     frames_per_window = int(fps * config.window_seconds)
#     print(f"5秒窗口对应的帧数：{frames_per_window}帧 (FPS={fps:.2f})")
    
#     # 计算窗口数量和每个窗口的帧范围
#     windows = {}
#     window_idx = 0
#     start_frame = 0
    
#     while start_frame < total_frames:
#         # 计算当前窗口的结束帧
#         end_frame = min(start_frame + frames_per_window, total_frames)
        
#         # 计算窗口的时间范围（秒）
#         start_time = start_frame / fps
#         end_time = end_frame / fps
        
#         # 提取当前窗口的采样索引（均匀采样）
#         sample_indices = list(range(start_frame, end_frame, config.sample_rate_per_window))[:config.max_frames_per_window]
#         print(f"窗口{window_idx}：时间范围[{start_time:.2f}, {end_time:.2f}]秒，采样帧数={len(sample_indices)}")
        
#         # 提取当前窗口的帧
#         window_frames = []
#         for idx in sample_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             success, frame = cap.read()
#             if success:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 pil_frame = Image.fromarray(frame_rgb)
#                 window_frames.append(pil_frame)
#             else:
#                 print(f"警告：窗口{window_idx}的帧{idx}读取失败，跳过")
        
#         # 兜底：如果窗口内帧数过少，补最后一帧
#         if len(window_frames) < 1:
#             window_frames = [Image.new("RGB", (config.clip_image_size, config.clip_image_size))]
#         elif len(window_frames) < 5:
#             last_frame = window_frames[-1]
#             while len(window_frames) < 5:
#                 window_frames.append(last_frame)
        
#         # 保存当前窗口信息
#         windows[window_idx] = {
#             "frames": window_frames,
#             "start_time": start_time,
#             "end_time": end_time,
#             "start_frame": start_frame,
#             "end_frame": end_frame
#         }
        
#         # 移动到下一个窗口
#         start_frame = end_frame
#         window_idx += 1
    
#     cap.release()
#     print(f"视频共切分为{len(windows)}个5秒窗口（最后一个窗口可能不足5秒）")
#     return windows

# # ===================== 提取单个窗口的512维特征 =====================
# def extract_window_512d_features(window_frames, config, model, preprocess, device):
#     """
#     提取单个5秒窗口的512维CLIP特征
#     """
#     # 预处理当前窗口的帧
#     processed_frames = []
#     for frame in window_frames:
#         frame_resized = frame.resize((config.clip_image_size, config.clip_image_size), Image.Resampling.LANCZOS)
#         processed = preprocess(frame_resized).unsqueeze(0)
#         processed_frames.append(processed)
    
#     # 拼接为批次并提取特征
#     batch = torch.cat(processed_frames).to(device)
#     model.eval()
    
#     with torch.no_grad():
#         features = model.encode_image(batch)
#         # 验证维度并归一化
#         assert features.shape[1] == config.feature_dim, f"特征维度错误：{features.shape[1]} != 512"
#         features = features / features.norm(dim=-1, keepdim=True)
    
#     # 转换为numpy数组
#     features_np = features.cpu().numpy().astype(config.save_dtype)
#     return features_np

# # ===================== 保存所有窗口的特征 =====================
# def save_window_features(all_window_features, video_name, save_dir, config):
#     """
#     保存每个窗口的特征，并生成汇总文件
#     :param all_window_features: 字典 {窗口编号: 特征矩阵}
#     :param video_name: 视频名称（用于命名文件）
#     :param save_dir: 保存目录
#     """
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 保存每个窗口的特征
#     window_info = {}
#     for window_idx, features in all_window_features.items():
#         # 构造文件名：视频名_窗口编号_5s.npy
#         save_path = os.path.join(save_dir, f"{video_name}_window_{window_idx}_5s.npy")
        
#         # 检查覆盖
#         if os.path.exists(save_path) and not config.overwrite:
#             raise FileExistsError(f"文件已存在：{save_path}，如需覆盖请设置overwrite=True")
        
#         # 保存特征
#         np.save(save_path, features)
#         window_info[window_idx] = {
#             "save_path": save_path,
#             "feature_shape": features.shape,
#             "num_frames": features.shape[0],
#             "feature_dim": config.feature_dim
#         }
#         print(f"窗口{window_idx}特征已保存：{save_path} (形状：{features.shape})")
    
#     # 保存汇总信息（可选）
#     summary_path = os.path.join(save_dir, f"{video_name}_windows_summary.npy")
#     np.save(summary_path, window_info)
#     print(f"窗口汇总信息已保存：{summary_path}")
    
#     return window_info

# # ===================== 主函数（按5秒窗口提取特征） =====================
# def extract_5s_window_features(video_path, save_dir):
#     """
#     主函数：按5秒窗口提取视频的512维CLIP特征
#     """
#     # 1. 初始化配置
#     config = CLIPVideoFeatureConfig()
    
#     # 2. 设置设备
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"\n使用设备：{device}")
    
#     # 3. 加载CLIP模型（只需加载一次，避免重复加载）
#     print("加载CLIP模型 (ViT-B/32) ...")
#     model, preprocess = clip.load(config.clip_model_name, device=device)
    
#     # 4. 切分视频为5秒窗口并提取帧
#     print("\n开始切分视频为5秒窗口...")
#     windows = split_video_into_5s_windows(video_path, config)
    
#     # 5. 逐窗口提取512维特征
#     all_window_features = {}
#     for window_idx, window_data in tqdm(windows.items(), desc="提取各窗口特征"):
#         frames = window_data["frames"]
#         # 提取当前窗口特征
#         window_features = extract_window_512d_features(frames, config, model, preprocess, device)
#         all_window_features[window_idx] = window_features
#         print(f"窗口{window_idx}：特征形状={window_features.shape} (帧数 × 512)")
    
#     # 6. 保存所有窗口特征
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     window_info = save_window_features(all_window_features, video_name, save_dir, config)
    
#     # 返回结果
#     return {
#         "total_windows": len(windows),
#         "window_features": all_window_features,
#         "window_info": window_info,
#         "config": config.__dict__
#     }

# # ===================== 测试与使用示例 =====================
# if __name__ == "__main__":
#     # 配置路径
#     VIDEO_PATH = "/data1/lihenghao/code/VadCLIP/TestData/video/Abuse001_x264.mp4"  # 替换为你的视频路径
#     SAVE_DIR = "/data1/lihenghao/code/VadCLIP/TestData/feature"  # 特征保存目录
    
#     # 执行5秒窗口特征提取
#     result = extract_5s_window_features(VIDEO_PATH, SAVE_DIR)
    
#     # 输出最终结果
#     print("\n=== 5秒窗口特征提取完成 ===")
#     print(f"总窗口数：{result['total_windows']}")
#     print(f"每个窗口特征维度：512维")
#     print(f"特征保存目录：{SAVE_DIR}")
    
#     # 示例：加载第一个窗口的特征
#     first_window_path = result["window_info"][0]["save_path"]
#     first_window_feat = np.load(first_window_path)
#     print(f"\n第一个窗口特征示例：")
#     print(f"形状：{first_window_feat.shape}")
#     print(f"均值：{np.mean(first_window_feat):.6f}")
#     print(f"模长均值：{np.mean(np.linalg.norm(first_window_feat, axis=1)):.4f} (应接近1)")
# %%%%%%%%%%%%%%%%%%%%%
# import os
# import cv2
# import torch
# import numpy as np
# import clip
# from PIL import Image
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")

# # ===================== 精细化配置（支持可选参数） =====================
# class CLIPVideoFeatureConfig:
#     """针对512维CLIP特征 + 可配置窗口的配置类"""
#     def __init__(self, window_seconds=5, max_frames_per_window=50):
#         # CLIP模型配置（固定ViT-B/32以保证512维）
#         self.clip_model_name = "ViT-B/32"
#         self.feature_dim = 512  # 固定512维
        
#         # 可配置的核心参数（可选参数）
#         self.window_seconds = window_seconds  # 窗口长度（秒），默认5秒
#         self.sample_rate_per_window = 2       # 每个窗口内的采样率（每隔2帧取1帧）
#         self.max_frames_per_window = max_frames_per_window  # 每个窗口最大抽帧数量，默认50
        
#         # 预处理配置（严格匹配CLIP的要求）
#         self.clip_image_size = 224  # ViT-B/32要求的输入尺寸
#         self.clip_mean = (0.48145466, 0.4578275, 0.40821073)  # CLIP官方均值
#         self.clip_std = (0.26862954, 0.26130258, 0.27577711)   # CLIP官方标准差
        
#         # 保存配置
#         self.save_dtype = np.float32  # 512维特征用float32足够，节省空间
#         self.overwrite = False        # 不覆盖已存在的npy文件

# # ===================== 参数校验函数 =====================
# def validate_params(window_seconds, max_frames_per_window):
#     """校验窗口长度和抽帧数量的合法性"""
#     if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
#         raise ValueError(f"窗口长度必须是大于0的数值，当前值：{window_seconds}")
#     if not isinstance(max_frames_per_window, int) or max_frames_per_window <= 0:
#         raise ValueError(f"每个窗口抽帧数量必须是大于0的整数，当前值：{max_frames_per_window}")
#     print(f"参数校验通过：窗口长度={window_seconds}秒，每个窗口最大抽帧数={max_frames_per_window}")

# # ===================== 按可配置窗口提取帧 =====================
# def split_video_into_windows(video_path, config):
#     """
#     将视频切分为可配置时长的窗口，并提取每个窗口的帧
#     :return: 字典 {窗口编号: (帧列表, 窗口起始时间, 窗口结束时间)}
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"视频文件无法打开: {video_path}")
    
#     # 获取视频基础信息
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     duration = total_frames / fps if fps > 0 else 0
#     print(f"视频基础信息：总帧数={total_frames}, FPS={fps:.2f}, 总时长={duration:.2f}秒")
    
#     # 计算窗口对应的帧数
#     frames_per_window = int(fps * config.window_seconds)
#     print(f"{config.window_seconds}秒窗口对应的帧数：{frames_per_window}帧 (FPS={fps:.2f})")
    
#     # 计算窗口数量和每个窗口的帧范围
#     windows = {}
#     window_idx = 0
#     start_frame = 0
    
#     while start_frame < total_frames:
#         # 计算当前窗口的结束帧
#         end_frame = min(start_frame + frames_per_window, total_frames)
        
#         # 计算窗口的时间范围（秒）
#         start_time = start_frame / fps
#         end_time = end_frame / fps
        
#         # 提取当前窗口的采样索引（均匀采样，不超过最大抽帧数量）
#         sample_indices = list(range(start_frame, end_frame, config.sample_rate_per_window))[:config.max_frames_per_window]
#         print(f"窗口{window_idx}：时间范围[{start_time:.2f}, {end_time:.2f}]秒，采样帧数={len(sample_indices)}")
        
#         # 提取当前窗口的帧
#         window_frames = []
#         for idx in sample_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             success, frame = cap.read()
#             if success:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 pil_frame = Image.fromarray(frame_rgb)
#                 window_frames.append(pil_frame)
#             else:
#                 print(f"警告：窗口{window_idx}的帧{idx}读取失败，跳过")
        
#         # 兜底：如果窗口内帧数过少，补最后一帧
#         if len(window_frames) < 1:
#             window_frames = [Image.new("RGB", (config.clip_image_size, config.clip_image_size))]
#         elif len(window_frames) < 5:
#             last_frame = window_frames[-1]
#             while len(window_frames) < 5:
#                 window_frames.append(last_frame)
        
#         # 保存当前窗口信息
#         windows[window_idx] = {
#             "frames": window_frames,
#             "start_time": start_time,
#             "end_time": end_time,
#             "start_frame": start_frame,
#             "end_frame": end_frame
#         }
        
#         # 移动到下一个窗口
#         start_frame = end_frame
#         window_idx += 1
    
#     cap.release()
#     print(f"视频共切分为{len(windows)}个{config.window_seconds}秒窗口（最后一个窗口可能不足）")
#     return windows

# # ===================== 提取单个窗口的512维特征 =====================
# def extract_window_512d_features(window_frames, config, model, preprocess, device):
#     """
#     提取单个窗口的512维CLIP特征
#     """
#     # 预处理当前窗口的帧
#     processed_frames = []
#     for frame in window_frames:
#         frame_resized = frame.resize((config.clip_image_size, config.clip_image_size), Image.Resampling.LANCZOS)
#         processed = preprocess(frame_resized).unsqueeze(0)
#         processed_frames.append(processed)
    
#     # 拼接为批次并提取特征
#     batch = torch.cat(processed_frames).to(device)
#     model.eval()
    
#     with torch.no_grad():
#         features = model.encode_image(batch)
#         # 验证维度并归一化
#         assert features.shape[1] == config.feature_dim, f"特征维度错误：{features.shape[1]} != 512"
#         features = features / features.norm(dim=-1, keepdim=True)
    
#     # 转换为numpy数组
#     features_np = features.cpu().numpy().astype(config.save_dtype)
#     return features_np

# # ===================== 保存所有窗口的特征 =====================
# def save_window_features(all_window_features, video_name, save_dir, config):
#     """
#     保存每个窗口的特征，并生成汇总文件
#     """
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 保存每个窗口的特征
#     window_info = {}
#     for window_idx, features in all_window_features.items():
#         # 构造文件名：视频名_窗口编号_窗口时长s.npy
#         save_path = os.path.join(save_dir, f"{video_name}_window_{window_idx}_{config.window_seconds}s.npy")
        
#         # 检查覆盖
#         if os.path.exists(save_path) and not config.overwrite:
#             raise FileExistsError(f"文件已存在：{save_path}，如需覆盖请设置overwrite=True")
        
#         # 保存特征
#         np.save(save_path, features)
#         window_info[window_idx] = {
#             "save_path": save_path,
#             "feature_shape": features.shape,
#             "num_frames": features.shape[0],
#             "feature_dim": config.feature_dim,
#             "window_seconds": config.window_seconds
#         }
#         print(f"窗口{window_idx}特征已保存：{save_path} (形状：{features.shape})")
    
#     # 保存汇总信息（可选）
#     summary_path = os.path.join(save_dir, f"{video_name}_windows_summary_{config.window_seconds}s.npy")
#     np.save(summary_path, window_info)
#     print(f"窗口汇总信息已保存：{summary_path}")
    
#     return window_info

# # ===================== 主函数（支持可选参数） =====================
# def extract_window_features(
#     video_path, 
#     save_dir, 
#     window_seconds=5,  # 可选参数：窗口长度，默认5秒
#     max_frames_per_window=50,  # 可选参数：每个窗口最大抽帧数量，默认50
#     overwrite=False  # 可选参数：是否覆盖已有文件，默认False
# ):
#     """
#     主函数：按可配置窗口提取视频的512维CLIP特征
    
#     参数说明：
#     - video_path: 输入视频路径
#     - save_dir: 特征保存目录
#     - window_seconds: 窗口长度（秒），可选，默认5秒
#     - max_frames_per_window: 每个窗口最大抽帧数量，可选，默认50
#     - overwrite: 是否覆盖已有文件，可选，默认False
#     """
#     # 1. 参数校验
#     validate_params(window_seconds, max_frames_per_window)
    
#     # 2. 初始化配置（传入可选参数）
#     config = CLIPVideoFeatureConfig(
#         window_seconds=window_seconds,
#         max_frames_per_window=max_frames_per_window
#     )
#     config.overwrite = overwrite  # 设置覆盖参数
    
#     # 3. 设置设备
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"\n使用设备：{device}")
    
#     # 4. 加载CLIP模型（只需加载一次，避免重复加载）
#     print(f"加载CLIP模型 (ViT-B/32) ...")
#     model, preprocess = clip.load(config.clip_model_name, device=device)
    
#     # 5. 切分视频为指定时长窗口并提取帧
#     print(f"\n开始切分视频为{window_seconds}秒窗口...")
#     windows = split_video_into_windows(video_path, config)
    
#     # 6. 逐窗口提取512维特征
#     all_window_features = {}
#     for window_idx, window_data in tqdm(windows.items(), desc="提取各窗口特征"):
#         frames = window_data["frames"]
#         # 提取当前窗口特征
#         window_features = extract_window_512d_features(frames, config, model, preprocess, device)
#         all_window_features[window_idx] = window_features
#         print(f"窗口{window_idx}：特征形状={window_features.shape} (帧数 × 512)")
    
#     # 7. 保存所有窗口特征
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     window_info = save_window_features(all_window_features, video_name, save_dir, config)
    
#     # 返回结果
#     return {
#         "total_windows": len(windows),
#         "window_features": all_window_features,
#         "window_info": window_info,
#         "config": config.__dict__
#     }

# # ===================== 测试与使用示例 =====================
# if __name__ == "__main__":
#     # 配置路径
#     VIDEO_PATH = "/data1/lihenghao/code/VadCLIP/TestData/video/Abuse001_x264.mp4"  # 替换为你的视频路径
#     SAVE_DIR = "/data1/lihenghao/code/VadCLIP/TestData/feature" + os.path.basename(VIDEO_PATH).split(".")[0]
    
#     # 示例1：使用默认参数（5秒窗口，每个窗口最多50帧）
#     # print("=== 示例1：使用默认参数（5秒窗口，每个窗口最多50帧） ===")
#     # result1 = extract_window_features(VIDEO_PATH, SAVE_DIR)
    
#     # # 示例2：自定义参数（3秒窗口，每个窗口最多30帧，允许覆盖）
#     # print("\n=== 示例2：自定义参数（3秒窗口，每个窗口最多30帧） ===")
#     # result2 = extract_window_features(
#     #     video_path=VIDEO_PATH,
#     #     save_dir=SAVE_DIR,
#     #     window_seconds=3,
#     #     max_frames_per_window=30,
#     #     overwrite=True
#     # )
    
#     # 示例3：自定义参数（10秒窗口，每个窗口最多100帧）
#     print("\n=== 示例3：自定义参数（10秒窗口，每个窗口最多100帧） ===")
#     result3 = extract_window_features(
#         video_path=VIDEO_PATH,
#         save_dir=SAVE_DIR,
#         window_seconds=5,
#         max_frames_per_window=300
#     )
    
#     # 输出最终结果
#     print("\n=== 所有提取任务完成 ===")
#     # print(f"示例1（5秒窗口）：{result1['total_windows']}个窗口")
#     # print(f"示例2（3秒窗口）：{result2['total_windows']}个窗口")
#     # print(f"示例3（10秒窗口）：{result3['total_windows']}个窗口")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import cv2
import torch
import numpy as np
import clip
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class CLIPVideoFeatureConfig:
    """针对512维CLIP特征 + 可配置窗口的配置类"""
    def __init__(self, window_seconds=5, max_frames_per_window=50):
        # CLIP模型配置
        self.clip_model_name = "ViT-B/16"
        self.feature_dim = 512
        
        # 可配置核心参数
        self.window_seconds = window_seconds
        self.max_frames_per_window = max_frames_per_window
        self.sample_rate_per_window = 2  # 初始默认值，会动态调整
        
        # 预处理配置
        self.clip_image_size = 224
        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)
        
        # 保存配置
        self.save_dtype = np.float32
        self.overwrite = False

def validate_params(window_seconds, max_frames_per_window):
    """校验参数合法性"""
    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
        raise ValueError(f"窗口长度必须>0，当前：{window_seconds}")
    if not isinstance(max_frames_per_window, int) or max_frames_per_window <= 0:
        raise ValueError(f"抽帧数量必须>0，当前：{max_frames_per_window}")
    print(f"参数校验通过：窗口长度={window_seconds}秒，目标抽帧数={max_frames_per_window}")

def split_video_into_windows(video_path, config):
    """
    切分视频为指定窗口，动态调整采样率以接近目标抽帧数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    
    # 视频基础信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"视频信息：总帧数={total_frames}, FPS={fps:.2f}, 总时长={duration:.2f}秒")
    
    # 计算窗口总帧数
    frames_per_window = int(fps * config.window_seconds)
    print(f"{config.window_seconds}秒窗口总帧数：{frames_per_window}帧")
    
    # 动态调整采样率（核心修改！）
    # 目标：让采样帧数尽可能接近max_frames_per_window
    if frames_per_window >= config.max_frames_per_window:
        # 窗口帧数足够，计算采样率（向上取整）
        config.sample_rate_per_window = max(1, int(np.ceil(frames_per_window / config.max_frames_per_window)))
    else:
        # 窗口帧数不足，采样率设为1（每帧都取），并提示
        config.sample_rate_per_window = 1
        print(f"提示：{config.window_seconds}秒窗口仅{frames_per_window}帧，小于目标{config.max_frames_per_window}帧，将提取所有帧")
    
    print(f"动态调整采样率为：{config.sample_rate_per_window}（每隔{config.sample_rate_per_window}帧取1帧）")
    
    # 切分窗口并提取帧
    windows = {}
    window_idx = 0
    start_frame = 0
    
    while start_frame < total_frames:
        end_frame = min(start_frame + frames_per_window, total_frames)
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # 计算采样索引（按调整后的采样率）
        sample_indices = list(range(start_frame, end_frame, config.sample_rate_per_window))[:config.max_frames_per_window]
        actual_sample_num = len(sample_indices)
        print(f"窗口{window_idx}：时间[{start_time:.2f}, {end_time:.2f}]秒，实际采样帧数={actual_sample_num}")
        
        # 提取帧
        window_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window_frames.append(Image.fromarray(frame_rgb))
            else:
                print(f"警告：窗口{window_idx}帧{idx}读取失败")
        
        # 兜底补帧
        if len(window_frames) < 1:
            window_frames = [Image.new("RGB", (config.clip_image_size, config.clip_image_size))]
        elif len(window_frames) < 5:
            last_frame = window_frames[-1]
            while len(window_frames) < 5:
                window_frames.append(last_frame)
        
        windows[window_idx] = {
            "frames": window_frames,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame
        }
        
        start_frame = end_frame
        window_idx += 1
    
    cap.release()
    print(f"共切分{len(windows)}个{config.window_seconds}秒窗口")
    return windows

def extract_window_512d_features(window_frames, config, model, preprocess, device):
    """提取单个窗口的512维特征"""
    processed_frames = []
    for frame in window_frames:
        frame_resized = frame.resize((config.clip_image_size, config.clip_image_size), Image.Resampling.LANCZOS)
        processed = preprocess(frame_resized).unsqueeze(0)
        processed_frames.append(processed)
    
    batch = torch.cat(processed_frames).to(device)
    model.eval()
    
    with torch.no_grad():
        features = model.encode_image(batch)
        assert features.shape[1] == config.feature_dim, f"维度错误：{features.shape[1]} != 512"
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().astype(config.save_dtype)

def save_window_features(all_window_features, video_name, save_dir, config):
    """保存窗口特征"""
    os.makedirs(save_dir, exist_ok=True)
    window_info = {}
    
    for window_idx, features in all_window_features.items():
        save_path = os.path.join(save_dir, f"{video_name}_window_{window_idx}_{config.window_seconds}s.npy")
        if os.path.exists(save_path) and not config.overwrite:
            raise FileExistsError(f"文件已存在：{save_path}，设置overwrite=True可覆盖")
        
        np.save(save_path, features)
        window_info[window_idx] = {
            "save_path": save_path,
            "feature_shape": features.shape,
            "num_frames": features.shape[0],
            "feature_dim": config.feature_dim
        }
        print(f"窗口{window_idx}特征保存：{save_path} (形状：{features.shape})")
    
    summary_path = os.path.join(save_dir, f"{video_name}_windows_summary.npy")
    np.save(summary_path, window_info)
    print(f"汇总信息保存：{summary_path}")
    return window_info

def extract_window_features(
    video_path, 
    save_dir, 
    window_seconds=5,
    max_frames_per_window=50,
    overwrite=False
):
    """主函数：提取可配置窗口的视频特征"""
    # 1. 参数校验
    validate_params(window_seconds, max_frames_per_window)
    
    # 2. 初始化配置
    config = CLIPVideoFeatureConfig(
        window_seconds=window_seconds,
        max_frames_per_window=max_frames_per_window
    )
    config.overwrite = overwrite
    
    # 3. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备：{device}")
    
    # 4. 加载CLIP模型
    print(f"加载CLIP模型 (ViT-B/32) ...")
    model, preprocess = clip.load(config.clip_model_name, device=device)
    
    # 5. 切分窗口并提取帧（动态调整采样率）
    print(f"\n切分视频为{window_seconds}秒窗口...")
    windows = split_video_into_windows(video_path, config)
    
    # 6. 逐窗口提取特征
    all_window_features = {}
    for window_idx, window_data in tqdm(windows.items(), desc="提取特征"):
        frames = window_data["frames"]
        window_features = extract_window_512d_features(frames, config, model, preprocess, device)
        all_window_features[window_idx] = window_features
        print(f"窗口{window_idx}特征形状：{window_features.shape}")
    
    # 7. 保存特征
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    window_info = save_window_features(all_window_features, video_name, save_dir, config)
    
    return {
        "total_windows": len(windows),
        "window_features": all_window_features,
        "window_info": window_info,
        "config": config.__dict__
    }

# 测试示例
if __name__ == "__main__":
    VIDEO_PATH = "/data1/lihenghao/code/VadCLIP/TestData/video/Abuse001_x264.mp4"  # 替换为你的视频路径
    SAVE_DIR = "/data1/lihenghao/code/VadCLIP/TestData/feature4" + os.path.basename(VIDEO_PATH).split(".")[0]
    
    # 你的场景：5秒窗口，目标抽300帧
    result = extract_window_features(
        video_path=VIDEO_PATH,
        save_dir=SAVE_DIR,
        window_seconds=5,
        max_frames_per_window=300,
        overwrite=True
    )
    
    print(f"\n最终结果：共{result['total_windows']}个窗口，采样率={result['config']['sample_rate_per_window']}")