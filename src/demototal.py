import os
import cv2
import torch
import numpy as np
import clip
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 集成暴力检测函数 =====================
def detect_violence(feat_path, model_path, device='cuda'):
    """
    检测视频中是否存在暴力异常以及异常类型
    """
    # 加载视频特征
    visual_feat = np.load(feat_path)
    print(f"\n[检测] 加载特征: {feat_path}, 特征形状: {visual_feat.shape}")
    length = visual_feat.shape[0]
    
    # 导入并使用真实的combined_option配置
    import combined_option
    args, _ = combined_option.parser.parse_known_args()
    
    from model import CLIPVAD
    from utils.tools import get_batch_mask, get_prompt_text
    
    # 暴力类型标签映射
    label_map = {
        # UCF数据集标签
        'Normal': 'normal', 
        'Abuse': 'abuse', 
        'Arrest': 'arrest', 
        'Arson': 'arson', 
        'Assault': 'assault', 
        'Burglary': 'burglary', 
        'Explosion': 'explosion', 
        'Fighting': 'fighting', 
        'RoadAccidents': 'roadAccidents', 
        'Robbery': 'robbery', 
        'Shooting': 'shooting', 
        'Shoplifting': 'shoplifting', 
        'Stealing': 'stealing', 
        'Vandalism': 'vandalism',
        # XD数据集标签
        'Riot': 'riot'
    }
    
    # 获取提示文本
    prompt_text = get_prompt_text(label_map)
    
    # 初始化模型
    model = CLIPVAD(
        args.classes_num, 
        args.embed_dim, 
        args.visual_length, 
        args.visual_width, 
        args.visual_head, 
        args.visual_layers, 
        args.attn_window, 
        args.prompt_prefix, 
        args.prompt_postfix, 
        device
    )
    
    # 加载预训练权重
    model_param = torch.load(model_path, map_location=device)
    if 'model' in model_param:
        model_param = model_param['model']
    model.load_state_dict(model_param)
    model.to(device)
    model.eval()
    
    # 处理特征
    maxlen = args.visual_length
    len_cur = length
    
    # 特征分块处理
    if len_cur < maxlen:
        visual = np.pad(visual_feat, ((0, maxlen - len_cur), (0, 0)), mode='constant', constant_values=0)
        visual = torch.tensor(visual).unsqueeze(0).to(device)
        lengths = torch.tensor([len_cur]).to(device)
    else:
        num_blocks = int(len_cur / maxlen) + 1
        visual_blocks = []
        lengths_list = []
        
        for i in range(num_blocks):
            start = i * maxlen
            end = start + maxlen
            if end > len_cur:
                block = np.pad(visual_feat[start:], ((0, end - len_cur), (0, 0)), mode='constant', constant_values=0)
                lengths_list.append(len_cur - start)
            else:
                block = visual_feat[start:end]
                lengths_list.append(maxlen)
            visual_blocks.append(block)
        
        visual = torch.tensor(np.stack(visual_blocks)).to(device)
        lengths = torch.tensor(lengths_list).to(device)
    
    # 生成padding mask
    padding_mask = get_batch_mask(lengths, maxlen).to(device)
    
    # 运行推理
    with torch.no_grad():
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
        type_probs = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
    
    # 确定是否存在暴力异常
    avg_confidence = prob2.mean().item()
    threshold = 0.5
    has_violence = avg_confidence > threshold
    
    # 确定异常类型
    violence_type = []
    avg_type_probs = np.mean(type_probs, axis=0)
    for i in range(1, len(avg_type_probs)):
        if avg_type_probs[i] > 0.1:
            violence_type.append(prompt_text[i])
    
    return has_violence, violence_type, avg_confidence

# ===================== 2. 配置类 =====================
class CLIPVideoFeatureConfig:
    """配置类"""
    def __init__(self, window_seconds=5, max_frames_per_window=50):
        # CLIP模型配置
        self.clip_model_name = "ViT-B/16"
        self.feature_dim = 512
        
        # 窗口配置
        self.window_seconds = window_seconds
        self.max_frames_per_window = max_frames_per_window
        self.sample_rate_per_window = 2
        
        # 预处理配置
        self.clip_image_size = 224
        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)
        
        # 保存配置
        self.save_dtype = np.float32
        self.overwrite = False

# ===================== 3. 工具函数 =====================
def validate_params(window_seconds, max_frames_per_window):
    """参数校验"""
    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
        raise ValueError(f"窗口长度必须>0，当前：{window_seconds}")
    if not isinstance(max_frames_per_window, int) or max_frames_per_window <= 0:
        raise ValueError(f"抽帧数量必须>0，当前：{max_frames_per_window}")
    print(f"参数校验通过：窗口长度={window_seconds}秒，目标抽帧数={max_frames_per_window}")

def split_video_into_windows(video_path, config):
    """切分视频为窗口，返回窗口信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    
    # 视频基础信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"\n视频基础信息：总帧数={total_frames}, FPS={fps:.2f}, 总时长={duration:.2f}秒")
    
    # 动态调整采样率
    frames_per_window = int(fps * config.window_seconds)
    print(f"{config.window_seconds}秒窗口总帧数：{frames_per_window}帧")
    
    if frames_per_window >= config.max_frames_per_window:
        config.sample_rate_per_window = max(1, int(np.ceil(frames_per_window / config.max_frames_per_window)))
    else:
        config.sample_rate_per_window = 1
        print(f"提示：窗口帧数不足，采样率设为1（提取所有帧）")
    
    print(f"最终采样率：每隔{config.sample_rate_per_window}帧取1帧")
    
    # 生成窗口列表
    windows = []
    window_idx = 0
    start_frame = 0
    
    while start_frame < total_frames:
        end_frame = min(start_frame + frames_per_window, total_frames)
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # 采样索引
        sample_indices = list(range(start_frame, end_frame, config.sample_rate_per_window))[:config.max_frames_per_window]
        
        # 提取帧
        window_frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window_frames.append(Image.fromarray(frame_rgb))
        
        # 兜底补帧
        if len(window_frames) < 1:
            window_frames = [Image.new("RGB", (config.clip_image_size, config.clip_image_size))]
        elif len(window_frames) < 5:
            last_frame = window_frames[-1]
            while len(window_frames) < 5:
                window_frames.append(last_frame)
        
        windows.append({
            "idx": window_idx,
            "frames": window_frames,
            "start_time": start_time,
            "end_time": end_time,
            "sample_num": len(window_frames)
        })
        
        start_frame = end_frame
        window_idx += 1
    
    cap.release()
    print(f"视频共切分为{len(windows)}个{config.window_seconds}秒窗口")
    return windows

def extract_single_window_features(window_frames, config, model, preprocess, device):
    """提取单个窗口的512维特征"""
    # 预处理帧
    processed_frames = []
    for frame in window_frames:
        frame_resized = frame.resize((config.clip_image_size, config.clip_image_size), Image.Resampling.LANCZOS)
        processed = preprocess(frame_resized).unsqueeze(0)
        processed_frames.append(processed)
    
    # 提取特征
    batch = torch.cat(processed_frames).to(device)
    model.eval()
    
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().astype(config.save_dtype)

def save_single_window_features(features, window_idx, video_name, save_dir, config, overwrite=False):
    """保存单个窗口的特征"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{video_name}_window_{window_idx}_{config.window_seconds}s.npy")
    
    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f"文件已存在：{save_path}，设置overwrite=True可覆盖")
    
    np.save(save_path, features)
    print(f"[保存] 窗口{window_idx}特征已保存至：{save_path}")
    return save_path

# ===================== 4. 主流程函数（逐窗口处理+检测） =====================
def process_video_window_by_window(
    video_path, 
    save_dir, 
    violence_model_path,  # 暴力检测模型路径
    window_seconds=5,
    max_frames_per_window=50,
    overwrite=False,
    detect_device='cuda'
):
    """
    核心函数：逐窗口处理视频
    流程：处理窗口0 → 提取特征 → 保存特征 → 暴力检测 → 处理窗口1 → ...
    """
    # 1. 初始化配置
    validate_params(window_seconds, max_frames_per_window)
    config = CLIPVideoFeatureConfig(window_seconds, max_frames_per_window)
    config.overwrite = overwrite
    
    # 2. 设备配置
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n\n======== 初始化 ========")
    print(f"CLIP运行设备：{clip_device}")
    print(f"检测模型运行设备：{detect_device}")
    
    # 3. 加载CLIP模型（只需加载一次）
    print(f"加载CLIP模型 ({config.clip_model_name}) ...")
    clip_model, clip_preprocess = clip.load(config.clip_model_name, device=clip_device)
    
    # 4. 切分视频为窗口
    windows = split_video_into_windows(video_path, config)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 5. 逐窗口处理（核心流程）
    all_detection_results = []  # 保存所有窗口的检测结果
    
    for window in tqdm(windows, desc="逐窗口处理"):
        window_idx = window["idx"]
        window_frames = window["frames"]
        start_time = window["start_time"]
        end_time = window["end_time"]
        
        print(f"\n\n=== 处理窗口{window_idx} [{start_time:.2f} - {end_time:.2f}秒] ===")
        
        # Step 1: 提取当前窗口特征
        print(f"[提取特征] 窗口{window_idx}，帧数={len(window_frames)}")
        window_features = extract_single_window_features(
            window_frames, config, clip_model, clip_preprocess, clip_device
        )
        print(f"[提取特征] 特征形状：{window_features.shape}")
        
        # Step 2: 保存当前窗口特征
        feat_save_path = save_single_window_features(
            window_features, window_idx, video_name, save_dir, config, overwrite
        )
        
        # Step 3: 执行暴力检测
        print(f"[暴力检测] 开始检测窗口{window_idx}...")
        try:
            has_violence, violence_type, confidence = detect_violence(
                feat_save_path, violence_model_path, detect_device
            )
        except Exception as e:
            print(f"[检测错误] 窗口{window_idx}检测失败：{str(e)}")
            has_violence = False
            violence_type = []
            confidence = 0.0
        
        # Step 4: 输出当前窗口检测结果
        print(f"\n[检测结果] 窗口{window_idx}：")
        print(f"  是否存在暴力：{'是' if has_violence else '否'}")
        print(f"  检测置信度：{confidence:.4f}")
        if has_violence:
            print(f"  暴力类型：{', '.join(violence_type) if violence_type else '未明确'}")
        
        # Step 5: 记录结果
        all_detection_results.append({
            "window_idx": window_idx,
            "time_range": (start_time, end_time),
            "feat_path": feat_save_path,
            "has_violence": has_violence,
            "violence_type": violence_type,
            "confidence": confidence
        })
    
    # 6. 输出最终汇总报告
    print(f"\n\n=== 所有窗口处理完成 ===")
    print(f"总窗口数：{len(windows)}")
    print(f"检测到暴力的窗口数：{sum(1 for res in all_detection_results if res['has_violence'])}")
    
    # 打印详细汇总
    print(f"\n=== 检测结果汇总 ===")
    for res in all_detection_results:
        time_range = f"{res['time_range'][0]:.2f} - {res['time_range'][1]:.2f}秒"
        violence_str = "有" if res['has_violence'] else "无"
        type_str = ', '.join(res['violence_type']) if res['violence_type'] else "无"
        print(f"窗口{res['window_idx']} [{time_range}]：暴力={violence_str}，置信度={res['confidence']:.4f}，类型={type_str}")
    
    # 返回所有结果
    return {
        "video_path": video_path,
        "window_config": {
            "window_seconds": window_seconds,
            "max_frames_per_window": max_frames_per_window
        },
        "detection_results": all_detection_results
    }

# ===================== 5. 测试与使用示例 =====================
if __name__ == "__main__":
    # 配置参数（根据你的实际路径修改）
    VIDEO_PATH = "/data1/lihenghao/code/VadCLIP/TestData/video/Abuse023_x264.mp4"                # 输入视频路径
    VIOLENCE_MODEL_PATH = "/data1/lihenghao/code/VadCLIP/modelcom/combined_model.pth"   # 暴力检测模型权重路径
    WINDOW_SECONDS = 5                           # 窗口长度（秒）
    MAX_FRAMES_PER_WINDOW = 300                  # 每个窗口最大抽帧数
    SAVE_DIR = "/data1/lihenghao/code/VadCLIP/TestData/video_window_features" + os.path.basename(VIDEO_PATH).split(".")[0] + "_" + str(WINDOW_SECONDS) + "_" + str(MAX_FRAMES_PER_WINDOW)           # 特征保存目录
    OVERWRITE = True                             # 是否覆盖已有特征文件
    DETECT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 执行逐窗口处理+检测
    final_results = process_video_window_by_window(
        video_path=VIDEO_PATH,
        save_dir=SAVE_DIR,
        violence_model_path=VIOLENCE_MODEL_PATH,
        window_seconds=WINDOW_SECONDS,
        max_frames_per_window=MAX_FRAMES_PER_WINDOW,
        overwrite=OVERWRITE,
        detect_device=DETECT_DEVICE
    )
    
    # 可选：保存检测结果到文件
    result_save_path = os.path.join(SAVE_DIR, "detection_results.npy")
    np.save(result_save_path, final_results["detection_results"])
    print(f"\n检测结果已保存至：{result_save_path}")