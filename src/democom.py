import torch
import numpy as np
from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text
import combined_option

def detect_violence(feat_path, model_path, device='cuda'):
    """
    检测视频中是否存在暴力异常以及异常类型
    
    参数:
        feat_path: 视频特征的npy文件路径
        model_path: 预训练模型权重路径
        device: 运行设备，默认'cuda'
        
    返回:
        has_violence: 布尔值，表示是否检测到暴力
        violence_type: 检测到的暴力类型列表
        confidence: 检测置信度
    """
    # 加载视频特征
    visual_feat = np.load(feat_path)
    print(f"特征形状: {visual_feat.shape}")  # 添加这行查看完整维度
    length = visual_feat.shape[0]
    print(f"加载特征: {feat_path}, 长度: {length}")
    
    # 设置模型参数，使用parse_known_args()忽略未知参数
    args, _ = combined_option.parser.parse_known_args()
    
    # 暴力类型标签映射 (与联合训练时保持一致)
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
        'Riot': 'riot'  # 新增的XD标签
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
    
    # 检查是否是完整的checkpoint文件
    if 'model' in model_param:
        # 如果是checkpoint文件，提取model部分
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
        # 计算需要分成多少块
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
        
        # 处理输出
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        
        # 计算异常概率 (使用logits2的方式，与测试代码一致)
        prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
        
        # 计算每种异常类型的概率
        type_probs = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
    
    # 确定是否存在暴力异常
    # 计算平均异常概率
    avg_confidence = prob2.mean().item()
    # 设置阈值 (可根据实际情况调整)
    threshold = 0.5
    has_violence = avg_confidence > threshold
    
    # 确定异常类型
    violence_type = []
    # 对每个时间步的类型概率取平均
    avg_type_probs = np.mean(type_probs, axis=0)
    # 排除正常类型
    for i in range(1, len(avg_type_probs)):
        if avg_type_probs[i] > 0.1:  # 类型概率阈值
            violence_type.append(prompt_text[i])
    
    return has_violence, violence_type, avg_confidence

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("使用方法: python combined_demo.py <特征文件路径> <模型权重路径>")
        sys.exit(1)
    
    feat_path = sys.argv[1]
    model_path = sys.argv[2]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 运行检测
    has_violence, violence_type, confidence = detect_violence(feat_path, model_path, device)
    
    # 输出结果
    print("\n===== 暴力异常检测结果 =====")
    print(f"是否存在暴力异常: {'是' if has_violence else '否'}")
    print(f"检测置信度: {confidence:.4f}")
    
    if has_violence:
        print(f"检测到的暴力类型: {', '.join(violence_type) if violence_type else '未明确类型'}")
    else:
        print("未检测到暴力行为")

# 使用示例:
# python src/combined_demo.py /data1/lihenghao/data/XDTestClipFeatures/wangted.2008__#0-27-17_0-28-07_label_A__9.npy /data1/lihenghao/code/VadCLIP/modelcom/combined_model.pth
