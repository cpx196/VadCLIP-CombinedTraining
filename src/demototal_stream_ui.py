import os
import cv2
import torch
import numpy as np
import clip
from PIL import Image
from tqdm import tqdm
import warnings
import sys
import time
warnings.filterwarnings("ignore")

# ===================== 1. 集成暴力检测函数（复用原有逻辑，无修改） =====================
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

# ===================== 2. 配置类（新增步长参数） =====================
class CLIPVideoFeatureConfig:
    """配置类"""
    def __init__(self, window_seconds=5, step_seconds=1, max_frames_per_window=50):
        # CLIP模型配置
        self.clip_model_name = "ViT-B/16"
        self.feature_dim = 512
        
        # 窗口配置（新增步长参数）
        self.window_seconds = window_seconds  # 窗口总时长
        self.step_seconds = step_seconds      # 滑动步长（每次窗口后移的时长）
        self.max_frames_per_window = max_frames_per_window
        self.sample_rate_per_window = 2
        
        # 预处理配置
        self.clip_image_size = 224
        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)
        
        # 保存配置
        self.save_dtype = np.float32
        self.overwrite = False

# ===================== 3. 工具函数（适配滑动窗口） =====================
def validate_params(window_seconds, step_seconds, max_frames_per_window):
    """参数校验（新增步长校验）"""
    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
        raise ValueError(f"窗口长度必须>0，当前：{window_seconds}")
    if not isinstance(step_seconds, (int, float)) or step_seconds <= 0 or step_seconds > window_seconds:
        raise ValueError(f"步长必须>0且≤窗口长度，当前：{step_seconds}")
    if not isinstance(max_frames_per_window, int) or max_frames_per_window <= 0:
        raise ValueError(f"抽帧数量必须>0，当前：{max_frames_per_window}")
    print(f"参数校验通过：窗口长度={window_seconds}秒，步长={step_seconds}秒，目标抽帧数={max_frames_per_window}")

def extract_single_window_features(window_frames, config, model, preprocess, device):
    """提取单个窗口的512维特征（复用原有逻辑）"""
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
    """保存单个窗口的特征（复用原有逻辑）"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{video_name}_sliding_window_{window_idx}_{config.window_seconds}s_step{config.step_seconds}s.npy")
    
    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f"文件已存在：{save_path}，设置overwrite=True可覆盖")
    
    np.save(save_path, features)
    return save_path

# ===================== 新增：终端输出控制工具函数 =====================
def clear_terminal():
    """清空终端（跨平台兼容）"""
    os.system('cls' if os.name == 'nt' else 'clear')

def move_cursor_up(n_lines):
    """将终端光标上移n行"""
    sys.stdout.write(f"\033[{n_lines}A")
    sys.stdout.flush()

def erase_current_line():
    """清空当前行"""
    sys.stdout.write("\033[K")
    sys.stdout.flush()

def print_detection_panel(video_info, window_config, current_result, total_windows, processed_windows):
    """打印固定格式的检测面板（用于实时刷新）"""
    # 清空终端后重新打印面板
    clear_terminal()
    
    # 面板标题
    print("="*60)
    print(f"          视频暴力检测 - 滑动窗口实时监控")
    print("="*60)
    
    # 1. 视频基础信息
    print(f"\n📹 视频信息：")
    print(f"   路径：{video_info['path']}")
    print(f"   总帧数：{video_info['total_frames']} | FPS：{video_info['fps']:.2f} | 总时长：{video_info['duration']:.2f}秒")
    
    # 2. 窗口配置
    print(f"\n⚙️  窗口配置：")
    print(f"   窗口长度：{window_config['window_seconds']}秒 | 滑动步长：{window_config['step_seconds']}秒")
    print(f"   单窗口最大抽帧数：{window_config['max_frames_per_window']}")
    
    # 3. 处理进度
    progress = (processed_windows / total_windows) * 100 if total_windows > 0 else 0
    progress_bar = "█" * int(progress/2) + "░" * (50 - int(progress/2))
    print(f"\n📊 处理进度：")
    print(f"   已处理窗口：{processed_windows}/{total_windows} [{progress_bar}] {progress:.1f}%")
    
    # 4. 当前窗口检测结果
    print(f"\n🔍 当前窗口 ({current_result['sliding_window_idx']}) 检测结果：")
    print(f"   时间范围：{current_result['time_range'][0]:.2f} - {current_result['time_range'][1]:.2f}秒")
    print(f"   特征路径：{os.path.basename(current_result['feat_path'])}")
    print(f"   暴力检测：{'✅ 存在' if current_result['has_violence'] else '❌ 不存在'}")
    print(f"   检测置信度：{current_result['confidence']:.4f}")
    print(f"   暴力类型：{', '.join(current_result['violence_type']) if current_result['violence_type'] else '未识别'}")
    
    # 5. 累计统计
    violent_windows = sum(1 for res in current_result['all_results'] if res['has_violence'])
    print(f"\n📈 累计统计：")
    print(f"   累计检测窗口：{processed_windows} | 检测到暴力的窗口数：{violent_windows}")
    print(f"   暴力窗口占比：{(violent_windows/processed_windows)*100:.1f}%" if processed_windows > 0 else "   暴力窗口占比：0.0%")
    
    # 6. 提示
    print(f"\n💡 提示：按 Ctrl+C 终止检测，检测完成后自动输出汇总报告")
    print("="*60)

# ===================== 4. 核心滑动窗口处理函数（修改：实时刷新输出） =====================
def process_video_sliding_window(
    video_path, 
    save_dir, 
    violence_model_path,  # 暴力检测模型路径
    window_seconds=5,
    step_seconds=1,       # 滑动步长（核心新增参数）
    max_frames_per_window=50,
    overwrite=False,
    detect_device='cuda'
):
    """
    核心函数：滑动窗口处理视频流（新增实时刷新终端输出）
    """
    # 1. 初始化配置
    validate_params(window_seconds, step_seconds, max_frames_per_window)
    config = CLIPVideoFeatureConfig(window_seconds, step_seconds, max_frames_per_window)
    config.overwrite = overwrite
    
    # 2. 设备配置
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. 加载CLIP模型（只需加载一次）
    clip_model, clip_preprocess = clip.load(config.clip_model_name, device=clip_device)
    
    # 4. 初始化视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    
    # 视频基础信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    video_info = {
        "path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration
    }
    
    # 计算窗口/步长对应的帧数
    frames_per_window = int(fps * config.window_seconds)  # 窗口总帧数
    frames_per_step = int(fps * config.step_seconds)      # 步长对应帧数
    # 动态调整采样率（保证窗口内抽帧数不超过max_frames_per_window）
    if frames_per_window >= config.max_frames_per_window:
        config.sample_rate_per_window = max(1, int(np.ceil(frames_per_window / config.max_frames_per_window)))
    else:
        config.sample_rate_per_window = 1
    
    # 预计算总窗口数（用于进度显示）
    total_windows = int(np.ceil((duration - window_seconds) / step_seconds)) + 1 if duration > window_seconds else 1
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    all_detection_results = []  # 保存所有滑动窗口的检测结果
    sliding_window_idx = 0      # 滑动窗口计数
    
    # 5. 初始化滑动窗口缓存（存储当前窗口的帧）
    window_frames_cache = []
    current_frame_idx = 0       # 当前读取到的视频帧索引
    
    # 6. 滑动窗口核心逻辑
    print(f"\n初始化完成，开始处理视频...")
    time.sleep(0.5)  # 短暂延迟，让用户看到初始化提示
    
    while current_frame_idx < total_frames:
        # Step 1: 填充当前窗口的帧（直到窗口满 或 视频结束）
        while len(window_frames_cache) < frames_per_window and current_frame_idx < total_frames:
            # 读取帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            success, frame = cap.read()
            if not success:
                break
            
            # 按采样率抽取帧（避免窗口内帧过多）
            if current_frame_idx % config.sample_rate_per_window == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window_frames_cache.append(Image.fromarray(frame_rgb))
            
            current_frame_idx += 1
        
        # 兜底补帧（保证窗口有至少1帧）
        if len(window_frames_cache) < 1:
            window_frames_cache = [Image.new("RGB", (config.clip_image_size, config.clip_image_size))]
        elif len(window_frames_cache) < 5:
            last_frame = window_frames_cache[-1]
            while len(window_frames_cache) < 5:
                window_frames_cache.append(last_frame)
        
        # 计算当前窗口的时间范围
        window_start_time = (current_frame_idx - len(window_frames_cache)*config.sample_rate_per_window) / fps
        window_end_time = window_start_time + config.window_seconds
        window_end_time = min(window_end_time, duration)  # 避免超过视频总时长
        
        # Step 2: 提取当前窗口特征
        window_features = extract_single_window_features(
            window_frames_cache, config, clip_model, clip_preprocess, clip_device
        )
        
        # Step 3: 保存当前窗口特征
        feat_save_path = save_single_window_features(
            window_features, sliding_window_idx, video_name, save_dir, config, overwrite
        )
        
        # Step 4: 执行暴力检测
        try:
            has_violence, violence_type, confidence = detect_violence(
                feat_save_path, violence_model_path, detect_device
            )
        except Exception as e:
            print(f"\n[检测错误] 滑动窗口{sliding_window_idx}检测失败：{str(e)}")
            has_violence = False
            violence_type = []
            confidence = 0.0
        
        # Step 5: 记录结果
        current_result = {
            "sliding_window_idx": sliding_window_idx,
            "time_range": (window_start_time, window_end_time),
            "feat_path": feat_save_path,
            "has_violence": has_violence,
            "violence_type": violence_type,
            "confidence": confidence,
            "all_results": all_detection_results  # 传递累计结果用于统计
        }
        all_detection_results.append(current_result)
        
        # Step 6: 实时刷新终端面板
        print_detection_panel(
            video_info=video_info,
            window_config={
                "window_seconds": window_seconds,
                "step_seconds": step_seconds,
                "max_frames_per_window": max_frames_per_window
            },
            current_result=current_result,
            total_windows=total_windows,
            processed_windows=sliding_window_idx + 1
        )
        time.sleep(0.05)  # 短暂延迟，避免刷新过快
        
        # Step 7: 滑动窗口（核心操作）：砍掉前面step对应的帧，保留剩余帧
        frames_to_remove = min(int(frames_per_step / config.sample_rate_per_window), len(window_frames_cache))
        window_frames_cache = window_frames_cache[frames_to_remove:]
        sliding_window_idx += 1
    
    # 释放视频流
    cap.release()
    
    # 7. 检测完成：输出最终汇总报告
    clear_terminal()
    print("="*60)
    print(f"          视频暴力检测 - 最终汇总报告")
    print("="*60)
    
    # 基础信息
    print(f"\n📹 视频信息：{os.path.basename(video_path)}")
    print(f"   总时长：{duration:.2f}秒 | 处理窗口数：{sliding_window_idx} | 窗口配置：{window_seconds}秒窗口/{step_seconds}秒步长")
    
    # 统计结果
    violent_windows = sum(1 for res in all_detection_results if res['has_violence'])
    print(f"\n📊 检测统计：")
    print(f"   检测到暴力的窗口数：{violent_windows}/{sliding_window_idx} ({(violent_windows/sliding_window_idx)*100:.1f}%)")
    print(f"   最高置信度：{max([res['confidence'] for res in all_detection_results]):.4f}")
    
    # 详细结果（折叠显示，避免过长）
    print(f"\n🔍 详细检测结果（仅显示暴力窗口）：")
    violent_results = [res for res in all_detection_results if res['has_violence']]
    if violent_results:
        for res in violent_results:
            print(f"   窗口{res['sliding_window_idx']} [{res['time_range'][0]:.2f}-{res['time_range'][1]:.2f}秒] | 类型：{', '.join(res['violence_type'])} | 置信度：{res['confidence']:.4f}")
    else:
        print(f"   未检测到任何暴力行为")
    
    print(f"\n💾 结果保存：")
    result_save_path = os.path.join(save_dir, "sliding_detection_results.npy")
    np.save(result_save_path, all_detection_results)
    print(f"   详细结果已保存至：{result_save_path}")
    print("="*60)
    
    # 返回所有结果
    return {
        "video_path": video_path,
        "window_config": {
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "max_frames_per_window": max_frames_per_window
        },
        "detection_results": all_detection_results
    }

# ===================== 5. 测试与使用示例 =====================
if __name__ == "__main__":
    # 配置参数（根据你的实际路径修改）
    VIDEO_PATH = "/home/chenpengxu/VadCLIP/TestData/video/Arson029_x264.mp4"                # 输入视频路径
    VIOLENCE_MODEL_PATH = "/home/chenpengxu/VadCLIP/modelcom/combined_model.pth"   # 暴力检测模型权重路径
    WINDOW_SECONDS = 5                           # 窗口长度（秒）
    STEP_SECONDS = 1                             # 滑动步长（秒，核心新增）
    MAX_FRAMES_PER_WINDOW = 300                  # 每个窗口最大抽帧数
    SAVE_DIR = "/home/chenpengxu/VadCLIP/TestData/video_sliding_features/" + os.path.basename(VIDEO_PATH).split(".")[0] + f"_{WINDOW_SECONDS}s_step{STEP_SECONDS}s"  # 特征保存目录
    OVERWRITE = True                             # 是否覆盖已有特征文件
    DETECT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 执行滑动窗口处理+检测
    final_results = process_video_sliding_window(
        video_path=VIDEO_PATH,
        save_dir=SAVE_DIR,
        violence_model_path=VIOLENCE_MODEL_PATH,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS,
        max_frames_per_window=MAX_FRAMES_PER_WINDOW,
        overwrite=OVERWRITE,
        detect_device=DETECT_DEVICE
    )