import os
import sys

# 将 src 目录插入到路径最前面，确保优先加载本地代码
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJ_ROOT, 'src'))
sys.path.insert(0, PROJ_ROOT)

import cv2
import torch
import json
import re
import numpy as np
import clip
from PIL import Image
from collections import deque
import threading
import time
import tempfile
from http import HTTPStatus
import dashscope
from concurrent.futures import ThreadPoolExecutor

# ================= 日志配置 =================
class Logger(object):
    def __init__(self, filename="pipeline_log.txt", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 初始化日志 (保存在 VadCLIP/logs/ 目录下)
LOG_DIR = os.path.join(PROJ_ROOT, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_PATH = os.path.join(LOG_DIR, f"detection_{int(time.time())}.txt")
sys.stdout = Logger(LOG_PATH, sys.stdout)
sys.stderr = Logger(LOG_PATH, sys.stderr)

from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text
import combined_option

# ================= 配置区 =================
DASHSCOPE_API_KEY = "sk-6ae970647261445d973f0589543c92bc"
dashscope.api_key = DASHSCOPE_API_KEY

VIDEO_MODEL = "qwen3-vl-plus"
VIOLENCE_MODEL_PATH = "/home/chenpengxu/VadCLIP/modelcom/combined_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 新增：控制参数
VLM_TRIGGER_THRESHOLD = 4     # 连续命中多少次触发 VLM
VLM_COOLDOWN_SECONDS = 5      # VLM 触发冷却时间
ALARM_CONFIDENCE_THRESHOLD = 0.35 # 进一步调低阈值至 0.35，增强灵敏度
WINDOW_DURATION = 10           # 窗口长度 (秒)
STEP_DURATION = 1              # 步长长度 (秒)
LOCAL_PROB_THRESHOLD = 0.15     # 本地检测阈值

# ================= 1. 加载模型 =================
print("正在初始化深度检测模型 (VadCLIP)...")
# 获取默认参数
args, _ = combined_option.parser.parse_known_args()
label_map = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 
    'Vandalism': 'vandalism', 'Riot': 'riot'
}
PROMPT_TEXT = get_prompt_text(label_map)

# 初始化 VadCLIP
vad_model = CLIPVAD(
    args.classes_num, args.embed_dim, args.visual_length, 
    args.visual_width, args.visual_head, args.visual_layers, 
    args.attn_window, args.prompt_prefix, args.prompt_postfix, DEVICE
)
model_param = torch.load(VIOLENCE_MODEL_PATH, map_location=DEVICE)
if 'model' in model_param:
    model_param = model_param['model']
vad_model.load_state_dict(model_param)
vad_model.to(DEVICE)
vad_model.eval()

# 加载 CLIP (确保使用 src/clip)
clip_model, clip_preprocess = clip.load("ViT-B/16", device=DEVICE)

# 异步线程池
executor = ThreadPoolExecutor(max_workers=3)
GLOBAL_VLM_REQUEST_ID = 0  # 全局请求 ID，用于追踪和舍弃过时任务

# ================= 2. 核心功能函数 =================

def low_level_detect(frames_deque):
    """
    使用 VadCLIP 对滑动窗口内的帧进行异常概率检测
    针对长窗口 (如 20s) 进行了抽样优化以防 GPU OOM
    """
    total_frames = len(frames_deque)
    if total_frames < 5:
        return 0.0, "normal"

    # 针对长窗口进行抽样，最多提 32 帧特征 (VadCLIP 常见的处理长度)
    # 这既能涵盖长跨度，又能避免 600+ 帧导致的 OOM
    target_detect_frames = 32
    if total_frames > target_detect_frames:
        indices = np.linspace(0, total_frames - 1, target_detect_frames, dtype=int)
        sampled_frames = [frames_deque[i] for i in indices]
    else:
        sampled_frames = list(frames_deque)

    # 将 PIL 图像转换为 CLIP 特征
    processed_frames = []
    for img in sampled_frames:
        processed = clip_preprocess(img).unsqueeze(0)
        processed_frames.append(processed)
    
    batch = torch.cat(processed_frames).to(DEVICE)
    
    with torch.no_grad():
        # 1. 提取图像特征
        features = clip_model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        visual_feat = features.unsqueeze(0) # (1, T_sampled, 512)
        
        # 2. 准备 VadCLIP 输入
        T = visual_feat.shape[1]
        maxlen = args.visual_length
        
        if T < maxlen:
            # 填充到 maxlen
            padding = torch.zeros((1, maxlen - T, visual_feat.shape[2])).to(DEVICE)
            visual_feat_padded = torch.cat([visual_feat, padding], dim=1)
        else:
            # 如果超过了则裁剪
            visual_feat_padded = visual_feat[:, :maxlen, :]
            
        lengths = torch.tensor([min(T, maxlen)]).to(DEVICE)
        padding_mask = get_batch_mask(lengths, maxlen).to(DEVICE)
        
        # 3. 推理
        _, _, logits2 = vad_model(visual_feat_padded, padding_mask, PROMPT_TEXT, lengths)
        
        # 计算各类别概率
        # logits2 shape: (batch, T_padded, num_class)
        probs = logits2[0, :T].softmax(dim=-1)
        
        # 显式计算正常和异常的分数 (取窗口平均值)
        normal_score = torch.mean(probs[:, 0]).item()
        abnormal_score = torch.mean(1 - probs[:, 0]).item()
        
        # 获取异常类别及其平均概率
        labels = list(label_map.keys())[1:] # 跳过 Normal
        anomaly_probs = probs[:, 1:] # 去掉 normal 分数
        avg_probs = torch.mean(anomaly_probs, dim=0) # 窗口内各类别平均概率
        
        # 以平均概率最高的类别作为窗口预测类别，其平均值作为 global_max_prob
        max_avg_val, max_avg_idx = torch.max(avg_probs, dim=0)
        predicted_type = labels[max_avg_idx.item()]
        global_max_prob = max_avg_val.item() # 使用平均值替代峰值，确保与 Top-K 输出和阈值逻辑一致

        # 计算 Top-K 信息用于展示
        topk_vals, topk_indices = torch.topk(avg_probs, k=min(5, len(labels)))
        topk_info = [(labels[idx.item()], val.item()) for val, idx in zip(topk_vals, topk_indices)]
        
    return normal_score, abnormal_score, global_max_prob, predicted_type, topk_info

def get_anomaly_criteria(anomaly_type: str) -> str:
    """获取不同异常类型的判断依据（拷贝自 test.py）"""
    criteria_dict = {
        "abuse": """
【定义】
在家庭环境中针对儿童、老年人、妇女或其他家庭成员的真实暴力、虐待或欺凌行为

【判断依据】
  1. 人物动作特征：包含有力的殴打、推拉、拳击、掐脖子等实际暴力动作（不是轻轻推打或游戏动作）
  2. 受害者行为：家庭成员出现真实的防御、哭泣、求救或试图躲避的反应（不是演戏或游戏）
  3. 身体接触：强制性的攻击行为，导致可见的伤害或明显的痛苦表现
  4. 环境特征：发生在室内家庭空间（卧室、客厅、厨房等）
  5. 受害者身份：针对儿童、老年人或配偶等
  6. 后果迹象：可能看到真实的伤害痕迹（淤青、划伤、流血等）或受害者明显的创伤反应
  7. 区分原则：区分真实虐待和家庭成员的游戏、开玩笑等行为。如果行为缺乏实际伤害或受害者的真实恐惧反应，可能是游戏行为""",
        
        "arson": """
【定义】
在家庭内故意纵火烧毁财产的真实行为

【判断依据】
  1. 火焰特征：客厅、卧室、厨房等室内出现明显的真实火焰和烟雾
  2. 人物动作：人物有明确的点火、靠近火源、拿着点火工具等动作
  3. 燃烧对象：真实的家庭物品（家具、床上用品、地毯、窗帘等）着火
  4. 场景特征：明显的浓烟、烟雾报警器报警、火焰蔓延迹象
  5. 时间特征：通常在家庭成员活跃时段或睡眠时段发生
  6. 真实性判断：区分真实火灾和玩具、模型火焰或装饰性火源。真实火灾应显示明显的热、烟和蔓延""",
        
        "assault": """
【定义】
一人或多人对受害者的真实暴力攻击（被攻击者处于被动防御状态）

【判断依据】
  1. 攻击者行为：发起明显的、有力的殴打、拳击、踢打等暴力动作
  2. 被攻击者行为：被攻击者处于防御或受伤状态，而不是积极还击
  3. 身体接触方式：直接暴力身体接触，显示出真实的伤害意图和力度
  4. 人数关系：一个或多个攻击者对一个被攻击者
  5. 结果：被攻击者被打倒、退缩、哭泣或表现出真实的害怕反应
  6. 动作力度：应该是有力的、真实的攻击，而不是轻微的推搡或游戏动作
  7. 区分原则：区分真实攻击和家庭成员间的游戏推搡。真实攻击应显示明显的力度和伤害迹象""",
        
        "fighting": """
【定义】
两个或多个人之间真实的肢体冲突或互相攻击

【判断依据】
  1. 人数：至少两个人以上参与其中
  2. 互动方式：双方都有明显的攻击和反击动作（不是轻微的推搡或开玩笑）
  3. 动作特征：有力的拳击、踢腿、推搡、扭打等暴力动作
  4. 姿态：双方都处于对抗、防守或攻击姿态
  5. 持续性：多个相互攻击的动作序列，持续时间较长
  6. 力度判断：动作应该具有明显的力度和影响，而不是轻轻的游戏动作
  7. 区分原则：区分真实打架和家庭成员的打闹、玩耍。真实打架应该显示出真实的冲突和伤害风险""",
        
        "robbery": """
【定义】
通过直接的暴力、威胁或强制手段，抢夺他人的金钱、财物或贵重物品

【判断依据】
  1. 暴力或威胁行为：对受害者进行明显的暴力攻击或明确威胁
  2. 财物直接转移：在受害者面前强行夺取金钱、珠宝、手机等贵重物品
  3. 受害者反应：受害者表现出明显的恐惧、被迫交出财物或抵抗的反应
  4. 人物身份：通常是陌生人或入侵者
  5. 行为明显性：与盗窃不同，抢劫是公开的、对抗性的、直接面对受害者的
  6. 时间地点：可能在任何时间发生，取决于受害者和财物位置
  7. 与盗窃的区别：盗窃是隐蔽的、不被发现的；抢劫是公开的、暴力的、面对面的""",
        
        "shooting": """
【定义】
用真实枪支射击他人的行为（严格仅限真实枪支，不包括玩具枪、气枪、仿真枪等）

【判断依据】
  1. 枪支真实性（最关键）：必须清晰识别为真实枪支，具有以下特征：
     - 金属枪身和真实枪机制
     - 真实的枪口和准星
     - 完整的手枪或步枪结构
     - 不能是：玩具枪、塑料枪、气枪、BB枪、软弹枪、仿真枪、水枪、道具枪等
  2. 排除非真枪：以下情况应判定为 normal：
     - 任何玩具枪、气枪、气压式枪支
     - 模型枪或复制品（即使看起来逼真）
     - 水枪或其他非致命射击工具
     - 电影、视频制作中的表演性射击
  3. 射击动作：人物举真实枪支、瞄准、扣扳机的动作
  4. 枪声：真实枪声特征（尖锐、响亮、力度强劲的枪击声）：
     - 排除：鞭炮、爆竹声、烟火声、气枪的"啪啪"声、其他背景声音
  5. 被击中迹象：被射击者出现明显的反应（倒地、流血、尖叫等实际伤害表现）
  6. 现场痕迹：真实的弹孔、血迹或其他真实伤害迹象
  7. 核心原则：仅当明确识别为真实枪支+真实枪声+真实伤害结果，才判定为 abnormal
     - 如有任何怀疑不是真枪，应判定为 normal，置信度为 0
     - 不要基于"看起来像枪"就判定为 shooting，必须有多项证据确认真实性""",
        
        "stealing": """
【定义】
在未获得明确授权的情况下，人员将不属于自己的财产或贵重物品 from 原有位置取走，
并通过藏匿或携带的方式离开当前场景的行为。

【核心判定逻辑（必须满足至少一条）】
  A. 明确的物品转移链路：
     - 物品原本静止放置（桌面、柜子、床头等）
     - 人物接触并拿起该物品
     - 物品被放入口袋、衣物、包内，或被持续携带
     - 人物随后离开当前房间或画面

  B. 藏匿行为完成：
     - 将物品放入不可直接观察的位置（口袋、背包、衣物内）
     - 且该物品在后续画面中不再回到原位

【辅助判据（提高置信度）】
  1. 人物在拿取前后出现观察四周、短暂停顿等确认环境安全的行为
  2. 行为发生在非日常活动时段（夜间、他人不在场）
  3. 拿取对象为高价值或非日常消耗物品（钱包、首饰、电子设备）

【身份与场景说明】
  - 可为陌生人入侵偷窃，也可为家庭成员之间的未经授权偷窃
  - 不要求出现暴力、冲突或对抗行为

【排除条件（必须严格执行）】
  1. 正常取用日常用品（钥匙、手机、遥控器等）并持续使用
  2. 明确的共享物品、生活用品或合理使用行为
  3. 拿取后物品很快放回原位，未发生藏匿或带离

【与 normal 的关键区别】
  stealing 必须包含“物品从原位置消失并被人物带离或隐藏”的完成状态，
  而 normal 不包含该完成态。""",
        
        "burglary": """
【定义】
陌生人或入侵者非法进入家庭，进行盗窃或破坏家庭财产的行为

【判断依据】
  1. 入侵迹象：出现未授权的陌生人进入家庭，或有撬锁、破窗、翻越等非法进入方式的迹象
  2. 入侵者身份：明确的陌生人或未授权人员出现在家庭环境中
  3. 盗取或破坏行为：入侵者在家庭内盗取贵重物品、打开柜子/抽屉、或对家庭设施进行破坏
  4. 隐蔽活动：入侵者表现出偷偷摸摸、小心谨慎的行为（避免被发现）
  5. 时间特征：通常在家庭成员不在家、睡眠或不注意时段发生
  6. 入侵证据：门窗被撬、破坏或其他非法进入的明显迹象
  7. 与 robbery 的区别：burglary 强调未授权进入和隐蔽盗窃；robbery 强调当面暴力夺取
  8. 与 stealing 的区别：burglary 涉及非法进入家庭环境；stealing 可能是家庭成员内部的盗窃""",
        
        "explosion": """
【定义】
事物爆炸的破坏性事件（不包括人为纵火或引爆）

【判断依据】
  1. 爆炸迹象：可见闪光、爆炸、物体四散
  2. 烟雾和火焰：浓烟、火焰和破坏的迹象
  3. 结构损坏：建筑物或物体的破坏、坍塌或碎片
  4. 冲击波效应：可见冲击波、物体被吹动
  5. 人员反应：周围人员的逃离或躲避反应""",
        
        "vandalism": """
【定义】
故意破坏或损坏公共或私人财产（如涂鸦、损坏等）

【判断依据】
  1. 破坏对象：针对财产进行破坏（墙壁、车辆、设施等）
  2. 破坏方式：用工具、颜料、火焰等进行破坏或涂鸦
  3. 人物动作：用手、工具或其他物品进行损坏动作
  4. 财产损伤：可见的划痕、破洞、涂鸦或破碎
  5. 故意性：行为明确表现出故意的破坏意图""",

        "normal": """
【定义】
正常的家庭日常活动，不存在犯罪或安全隐患

【判断依据】
  1. 日常活动：家庭成员进行正常的居家活动（看电视、做家务、用餐、睡觉等）
  2. 正常互动：家庭成员之间进行正常交互和沟通
  3. 无暴力迹象：不存在殴打、威胁、冲突或家庭暴力行为
  4. 有序环境：家庭环境井然有序，无混乱、破坏或异常现象
  5. 安全状态：所有家庭成员正常活动，无人遭受威胁或伤害状态"""
    }
    return criteria_dict.get(anomaly_type.lower(), "未知异常类型，请基于常见的犯罪特征进行判断")

def build_vlm_prompt(anomaly_type, confidence):
    """构建中立客观的 VLM 提示词，仅提供判断标准供其参考"""
    criteria = get_anomaly_criteria(anomaly_type)
    identity_info = "{\"person_count\": \"未知\", \"role\": \"未知\"}"
    
    prompt = f"""
你是一个家庭安全监控系统的多模态分析助手。你需要对给定的视频片段进行客观分析，判断其中是否包含真实的犯罪或异常行为。

【监控场景】
家庭环境（卧室、客厅、厨房等），如果不是家庭环境，也请尝试客观地做出判断

【核心指令】
1. 尝试否定深度学习模型的初步判定，如果没有明确否定的元素，请支持其结论。
═══════════════════════════════════════════════════════════════

【参考信息：初步检测结果】
深度学习模型检测到的疑似异常：{anomaly_type}
该类型的判断依据（供参考）：
{criteria}

═══════════════════════════════════════════════════════════════

【分析任务】

请对视频进行详尽分析，并按顺序回答以下内容（必须使用中文）：

1. **视频内容综述**：
   - 详细描述视频中发生了什么（人物动作、互动、环境变化）。

2. **多维度检查**：
   - 检查视频中是否存在：家庭暴力(abuse)、人身攻击(assault)、打架冲突(fighting)、抢劫(robbery)、盗窃(stealing)、入室盗窃(burglary)、枪支伤害(shooting)、纵火(arson)、爆炸(explosion)或财产破坏(vandalism)。

3. **最终结论**：
   - **状态判定**：normal（正常）或 abnormal（存在确凿的犯罪或严重威胁行为）。
   - **报警置信度**：给出 0 到 1 之间的数值。
     - 0: 完全正常，无任何威胁。
     - 0.1-0.4: 有轻微可疑但不确定，或属于正常的偏激烈互动。
     - 0.5-0.7: 存在明显符合犯罪特征的行为证据。
     - 0.8-1.0: 极度危险，证据确凿。

请以 JSON 格式返回最终决策，格式如下：
{{
    "status": "normal/abnormal",
    "alarm_confidence": 0.X,
    "description": "视频内容的整体描述",
    "dl_validation": "对模型初步判定 {anomaly_type} 的验证结论",
    "other_crimes_detected": "是否发现了除 {anomaly_type} 之外的其他犯罪",
    "reason": "你做出当前判定和置信度的最终理由"
}}

【重要提醒】
- 严禁捏造事实：如果视频中没有看到明确的犯罪动作，请判定为 normal。
- JSON 格式必须严格正确，不要包含 Markdown 代码块外的任何文字。
"""
    return prompt.strip()

def vlm_verify_async(frames_list, anomaly_type, confidence, frame_range, fps, request_id):
    """
    抽帧并合成视频发送给 VLM 进行多模态确认（替代拼接图逻辑）
    """
    def task():
        start_f, end_f = frame_range
        start_sec = start_f / fps
        end_sec = end_f / fps
        print(f"\n[VLM] 触发验证任务 | ID: {request_id} | 类型: {anomaly_type} | 范围: {start_f}-{end_f} ({start_sec:.1f}s-{end_sec:.1f}s)")
        
        # 1. 直接使用全部帧合成视频 (不再抽帧，保留连贯性)
        total_frames = len(frames_list)
        temp_video = os.path.join(PROJ_ROOT, f"vlm_temp_{int(time.time())}.mp4")
        
        try:
            if not frames_list:
                print("[VLM 错误] 无可用帧，任务取消")
                return
            
            # 使用列表中的所有帧
            selected_frames = [np.array(f) for f in frames_list]
            h, w = selected_frames[0].shape[:2]
            
            # 视频编码配置
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 这里的 FPS 尽量维持原始或稍微降低以平衡质量与体积
            # 如果原始 FPS 太高，合成 1000 多帧可能会导致请求超时，这里建议使用原 FPS
            out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h)) 
            
            for f in selected_frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
        except Exception as fe:
            print(f"[VLM 错误] 视频合成失败: {fe}")
            return

        # 2. 构建提示词
        prompt = build_vlm_prompt(anomaly_type, confidence)
        
        print(f"[VLM] 正在调用 {VIDEO_MODEL} (ID: {request_id}, 片段: {start_f:04d}-{end_f:04d}, 共 {total_frames} 帧)...")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"video": f"file://{os.path.abspath(temp_video)}"},
                        {"text": prompt}
                    ]
                }
            ]
            response = dashscope.MultiModalConversation.call(
                model=VIDEO_MODEL,
                messages=messages
            )

            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content[0]['text']
                print(f"\n" + "="*50)
                print(f" VLM 分析报告 ({start_f}-{end_f} | {start_sec:.1f}s-{end_sec:.1f}s)")
                print("-" * 50)
                print(content)
                print("="*50)
                
                # 尝试解析并打印报警结果
                # 先尝试移除 Markdown 代码块标记包围
                json_str = content
                if "```json" in content:
                    json_str = content.split("```json")[-1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[-1].split("```")[0]
                
                json_match = re.search(r'(\{.*\})', json_str, re.DOTALL)
                if json_match:
                    clean_json = json_match.group(1)
                    # 处理一些常见的 VLM JSON 错误：如多余的换行符或未转义的引号（简单尝试）
                    try:
                        res_json = json.loads(clean_json)
                    except json.JSONDecodeError:
                        # 尝试进一步清洗：移除尾部逗号等
                        try:
                            # 移除 JSON 对象的最后一个逗号 (e.g., "key": "val", })
                            clean_json_2 = re.sub(r',\s*\}', '}', clean_json)
                            res_json = json.loads(clean_json_2)
                        except Exception:
                            print(f"[VLM 警告] JSON 自动修复失败，准备尝试基础解析...")
                            # 最后尝试提取关键字段（如果不符合 JSON 规范）
                            status_match = re.search(r'"status":\s*"(\w+)"', clean_json)
                            conf_match = re.search(r'"alarm_confidence":\s*([0-9.]+)', clean_json)
                            res_json = {
                                "status": status_match.group(1) if status_match else "normal",
                                "alarm_confidence": float(conf_match.group(1)) if conf_match else 0.0,
                                "reason": "解析原始 JSON 失败，这是提取的备选值"
                            }

                    try:
                        status = res_json.get('status', 'normal').lower()
                        conf = float(res_json.get('alarm_confidence', 0))
                        
                        # --- VLM 置信度分级决策逻辑 ---
                        if conf < 0.2:
                            final_decision = "驳回DL请求，判定为正常"
                            decision_label = "【正常 / 忽略】"
                        elif 0.2 <= conf < 0.5:
                            final_decision = "通知相关人员：发现疑似异常"
                            decision_label = "【通知 / 关注】"
                        else: # conf >= 0.5
                            final_decision = "立刻发起报警：确认真实威胁！"
                            decision_label = "【警报 / 紧急】"

                        print(f"\n>>>> 最终决策: {decision_label}")
                        print(f"状态: {status} | 置信度: {conf:.2f}")
                        print(f"处理动作: {final_decision}")
                        print(f"判定原因: {res_json.get('reason', 'N/A')}")
                        print("="*50)
                        
                    except Exception as e:
                        print(f"[VLM 解析字段出错]: {e}")
                else:
                    print(f"[VLM] 未在返回内容中找到有效的 JSON 结构")
            else:
                print(f"[VLM] 错误: {response.code} - {response.message}")
        except Exception as e:
            print(f"[VLM] 调用异常: {e}")
        finally:
            if os.path.exists(temp_video):
                try: os.remove(temp_video)
                except: pass

    executor.submit(task)

# ================= 3. 主循环 =================

def run_pipeline(video_path):
    global GLOBAL_VLM_REQUEST_ID
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0 # 默认值防止除零

    window_size = int(WINDOW_DURATION * fps)
    window_step = int(STEP_DURATION * fps)

    # 滑动窗口队列，长度由时间决定
    frame_window = deque(maxlen=window_size)
    frame_count = 0
    consecutive_hits = 0
    last_vlm_time = 0
    last_vlm_prob = 0.0 # 记录最近一次触发 VLM 的置信度
    last_predicted_type = None

    # 新增：用于合并长视频片段的状态变量
    is_collecting_event = False
    event_start_frame = 0
    event_frames = [] # 存储事件期间的所有 PIL 图片
    event_max_prob = 0.0
    event_type = None

    print(f"\n开始播放与检测流... (视频 FPS: {fps:.1f}, 窗口: {WINDOW_DURATION}s, 步长: {STEP_DURATION}s, 按 'q' 键退出)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # 转换为 PIL Image 用于模型
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frame_window.append(pil_img)
        
        current_prob = 0.0
        normal_score = 1.0
        abnormal_score = 0.0
        predicted_type = "normal"
        topk_info = []
        
        # 每隔 window_step 帧检测一次 (滑动窗口检测，产生重叠)
        if frame_count >= window_size and frame_count % window_step == 0:
            normal_score, abnormal_score, current_prob, predicted_type, topk_info = low_level_detect(frame_window)
            
            # 确定当前窗口的帧范围
            end_f = frame_count
            start_f = max(1, frame_count - len(frame_window) + 1)
            
            # 判断是否命中异常 (新逻辑：低阈值 + 类别一致性)
            if current_prob >= LOCAL_PROB_THRESHOLD:
                if predicted_type == last_predicted_type:
                    consecutive_hits += 1
                else:
                    # 如果正在收集事件且类别变了，先结算前一个事件
                    if is_collecting_event:
                        print(f"  └─ [事件结束] 类别变更: {event_type} -> {predicted_type}. 发送 VLM 申请...")
                        GLOBAL_VLM_REQUEST_ID += 1
                        vlm_verify_async(event_frames, event_type, event_max_prob, (event_start_frame, end_f), fps, GLOBAL_VLM_REQUEST_ID)
                        is_collecting_event = False
                        event_frames = []
                    consecutive_hits = 1 # 类别改变，重新开始累计
                last_predicted_type = predicted_type
            else:
                # 置信度显著降低，如果正在收集则结算
                if is_collecting_event:
                    print(f"  └─ [事件结束] 置信度降低 ({current_prob:.2f} < {LOCAL_PROB_THRESHOLD}). 发送 VLM 申请...")
                    GLOBAL_VLM_REQUEST_ID += 1
                    vlm_verify_async(event_frames, event_type, event_max_prob, (event_start_frame, end_f), fps, GLOBAL_VLM_REQUEST_ID)
                    is_collecting_event = False
                    event_frames = []
                
                consecutive_hits = 0
                last_predicted_type = None
            
            # 详细打印检测状态
            if "DISPLAY" not in os.environ:
                start_sec = start_f / fps
                end_sec = end_f / fps
                type_msg = f"[{predicted_type}]" if consecutive_hits > 0 else "---"
                topk_str = ", ".join([f"{name}:{p:.2f}" for name, p in topk_info])
                print(f"[Window {start_f:04d}-{end_f:04d} | {start_sec:.1f}s-{end_sec:.1f}s] Normal: {normal_score:.3f} | Abnormal: {abnormal_score:.3f} | AvgProb: {current_prob:.2f} | Hits: {consecutive_hits}/{VLM_TRIGGER_THRESHOLD} {type_msg}")
                print(f"  └─ Top-K Categories: {topk_str}")
            
            # 逻辑变更：不再直接触发，而是开始/继续收集片段
            if consecutive_hits >= VLM_TRIGGER_THRESHOLD:
                if not is_collecting_event:
                    # 第一次达到触发阈值，开始收集
                    is_collecting_event = True
                    event_start_frame = start_f
                    event_type = predicted_type
                    event_max_prob = current_prob
                    # 获取当前窗口的所有帧作为起始
                    event_frames = list(frame_window)
                    print(f"  └─ [事件开始] 连续命中 {VLM_TRIGGER_THRESHOLD} 次，开始收集视频片段...")
                else:
                    # 已经在收集中，追加新采集的步长片段 (即最新的 window_step 帧)
                    # 注意：frame_window 包含整个 window_size，我们只需要最新的 step 部分
                    new_frames = list(frame_window)[-window_step:]
                    event_frames.extend(new_frames)
                    event_max_prob = max(event_max_prob, current_prob)
                    
                    # === 新增：时间窗口上限 (40s) 维护逻辑 ===
                    current_event_seconds = len(event_frames) / fps
                    if current_event_seconds >= 40:
                        print(f"  └─ [时间窗口达到上限] 40s 已到. 发送 VLM 申请并重新计时窗口...")
                        GLOBAL_VLM_REQUEST_ID += 1
                        # 结算当前 40s 片段
                        vlm_verify_async(event_frames, event_type, event_max_prob, (event_start_frame, end_f), fps, GLOBAL_VLM_REQUEST_ID)
                        
                        # 重置缓冲区，开始下一个 40s 片段（若异常仍在持续）
                        event_start_frame = end_f + 1
                        event_frames = [] # 重新开始缓冲帧
                        event_max_prob = 0.0 # 重置最大概率以便新窗口统计
                    # print(f"  └─ [继续收集] 当前事件总长度: {len(event_frames)} 帧")
        
        # 界面显示 (保持每帧更新)
        if "DISPLAY" in os.environ:
            display_frame = frame.copy()
            # 只要在收集中，就显示红色
            color = (0, 0, 255) if is_collecting_event else (0, 255, 0)
            
            start_f = max(1, frame_count - len(frame_window) + 1)
            start_sec = start_f / fps
            curr_sec = frame_count / fps
            
            cv2.putText(display_frame, f"Range: {start_f}-{frame_count} ({start_sec:.1f}s-{curr_sec:.1f}s)", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Normal: {normal_score:.2f} | Abnormal: {abnormal_score:.2f}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Avg Prob: {current_prob:.2f} ({predicted_type})", (30, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            status_text = f"Collecting Event ({event_type})" if is_collecting_event else f"Hits: {consecutive_hits}/{VLM_TRIGGER_THRESHOLD}"
            cv2.putText(display_frame, status_text, (30, 145), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            topk_text = " | ".join([f"{n}" for n, _ in topk_info[:3]])
            cv2.putText(display_frame, f"Top3: {topk_text}", (30, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            try:
                cv2.imshow("VadCLIP + VLM Pipeline", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass
        
        # time.sleep(0.01)

    cap.release()
    # 视频结束时，如果还有未结算的事件，进行最后一次提交
    if is_collecting_event and len(event_frames) > 0:
        print(f"  └─ [视频结束] 提交最后的事件片段...")
        GLOBAL_VLM_REQUEST_ID += 1
        vlm_verify_async(event_frames, event_type, event_max_prob, (event_start_frame, frame_count), fps, GLOBAL_VLM_REQUEST_ID)

    if "DISPLAY" in os.environ:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 视频搜索目录 (VadCLIP/test_video)
    SEARCH_DIR = os.path.join(PROJ_ROOT, "test_video")
    
    print("="*50)
    print(f" 视频异常检测系统已启动 (搜索目录: {SEARCH_DIR})")
    print("="*50)

    # 自动创建目录并列出可用视频
    if not os.path.exists(SEARCH_DIR):
        os.makedirs(SEARCH_DIR, exist_ok=True)
        print(f"[提示] 目录不存在，已创建。请将视频放入 {SEARCH_DIR}")
    
    videos = [f for f in os.listdir(SEARCH_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if videos:
        print("\n当前可用视频：")
        for i, v in enumerate(videos):
            print(f" {i+1}. {v}")
    else:
        print("\n[警告] 目录为空，请放入视频文件后再运行。")

    user_input = input("\n请输入视频编号或文件名 (输入 'q' 退出): ").strip()
    
    if user_input.lower() == 'q':
        sys.exit(0)
    
    # 处理编号输入
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(videos):
            video_path = os.path.join(SEARCH_DIR, videos[idx])
        else:
            print("编号超限。")
            sys.exit(1)
    else:
        # 直接按路径或文件名处理
        if os.path.isabs(user_input):
            video_path = user_input
        else:
            video_path = os.path.join(SEARCH_DIR, user_input)

    if os.path.exists(video_path):
        run_pipeline(video_path)
    else:
        print(f"\n[错误] 找不到视频文件: {video_path}")
