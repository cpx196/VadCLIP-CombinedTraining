import torch
import numpy as np
import os
from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义标签映射，包含暴力异常类型
label_map = dict({
    'A': 'normal', 
    'B1': 'fighting', 
    'B2': 'shooting', 
    'B4': 'riot', 
    'B5': 'abuse', 
    'B6': 'car accident', 
    'G': 'explosion'
})

# 定义模型参数（与训练时保持一致）
model_params = {
    'num_class': 7,  # 类别数：正常+6种异常
    'embed_dim': 512,
    'visual_length': 128,  # 每个块的最大长度
    'visual_width': 1024,  # 特征维度
    'visual_head': 8,
    'visual_layers': 6,
    'attn_window': 16,
    'prompt_prefix': 1,
    'prompt_postfix': 1,
    'device': device
}

def process_video_features(feat_path, maxlen):
    """处理视频特征，确保符合模型输入要求"""
    # 加载npy特征
    feat = np.load(feat_path)
    length = feat.shape[0]
    
    # 处理超过maxlen的情况
    if length > maxlen:
        split_num = int(length / maxlen) + 1
        split_feat = []
        for i in range(split_num):
            start_idx = i * maxlen
            end_idx = start_idx + maxlen
            if end_idx <= length:
                split_feat.append(feat[start_idx:end_idx])
            else:
                # 对最后一块进行补零
                pad_feat = np.pad(feat[start_idx:], ((0, end_idx - length), (0, 0)), 
                                 mode='constant', constant_values=0)
                split_feat.append(pad_feat)
        split_feat = np.stack(split_feat)
        return split_feat, length
    else:
        # 不足maxlen时补零
        pad_feat = np.pad(feat, ((0, maxlen - length), (0, 0)), 
                         mode='constant', constant_values=0)
        pad_feat = pad_feat.reshape(1, maxlen, -1)
        return pad_feat, length

def detect_violence(feat_path, weight_path, threshold=0.5):
    """
    检测视频中的暴力异常
    
    Args:
        feat_path: 视频npy特征路径
        weight_path: 训练权重路径
        threshold: 异常检测阈值
        
    Returns:
        has_violence: 是否存在暴力异常
        violence_type: 暴力异常类型（如果存在）
        confidence: 置信度
    """
    # 加载模型
    model = CLIPVAD(**model_params)
    model_param = torch.load(weight_path, map_location=device)
    model.load_state_dict(model_param)
    model.to(device)
    model.eval()
    
    # 获取文本提示
    prompt_text = get_prompt_text(label_map)
    
    # 处理视频特征
    visual, length = process_video_features(feat_path, model_params['visual_length'])
    visual = torch.tensor(visual, dtype=torch.float32).to(device)
    
    # 计算长度信息
    len_cur = length
    lengths = torch.zeros(int(length / model_params['visual_length']) + 1)
    remaining_length = length
    for j in range(len(lengths)):
        if remaining_length > model_params['visual_length']:
            lengths[j] = model_params['visual_length']
            remaining_length -= model_params['visual_length']
        else:
            lengths[j] = remaining_length
            break
    lengths = lengths.to(int).to(device)
    
    # 获取padding mask
    padding_mask = get_batch_mask(lengths, model_params['visual_length']).to(device)
    
    # 执行推理
    with torch.no_grad():
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
        
        # 处理logits1用于二分类（是否异常）
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
        
        # 处理logits2用于多分类（异常类型）
        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        prob2 = logits2[0:len_cur].softmax(dim=-1)
    
    # 分析结果
    avg_confidence = prob1.mean().item()  # 平均置信度
    has_violence = avg_confidence > threshold
    
    violence_type = "normal"
    if has_violence:
        # 获取最大概率的异常类型
        max_prob_class = prob2.mean(dim=0).argmax().item()
        violence_type = prompt_text[max_prob_class]
    
    return has_violence, violence_type, avg_confidence

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="暴力异常检测Demo")
    parser.add_argument("--feat_path", type=str, required=True, help="视频npy特征路径")
    parser.add_argument("--weight_path", type=str, required=True, help="训练权重路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="异常检测阈值")
    
    args = parser.parse_args()
    
    # 执行检测
    has_violence, violence_type, confidence = detect_violence(
        args.feat_path, args.weight_path, args.threshold
    )
    
    # 输出结果
    print("===== 暴力异常检测结果 =====")
    print(f"视频特征路径: {args.feat_path}")
    print(f"是否存在暴力异常: {'是' if has_violence else '否'}")
    print(f"暴力异常类型: {violence_type}")
    print(f"置信度: {confidence:.4f}")
    print(f"使用阈值: {args.threshold}")
    print("==========================")
