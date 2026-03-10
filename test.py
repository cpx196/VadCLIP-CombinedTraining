import dashscope
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ================= 配置区 =================
dashscope.api_key = "sk-6ae970647261445d973f0589543c92bc"

VIDEO_MODEL = "qwen3-vl-plus"
AUDIO_MODEL = "qwen3-omni-30b-a3b-captioner"

# ================= 数据模型 =================
class AnomalyDetectionPipeline:
    """异常检测完整流程"""
    
    def __init__(self):
        self.video_path = None
        self.identity_rules = {}
        self.anomaly_type_result = None  # 人工给出的异常类型
        self.vlm_result = None
    
    def set_video(self, video_path: str):
        """设置视频文件"""
        self.video_path = video_path
        print(f"视频已设置: {video_path}")
    
    def set_identity_from_rules(self, rules: Dict[str, Any]):
        """从规则引擎得到人物身份"""
        self.identity_rules = rules
        print(f"身份规则已设置: {rules}")
        return rules
    
    def set_anomaly_detection(self, anomaly_type: str, confidence: float):
        """
        设置异常检测网络的结果（暂时人工给出）
        
        Args:
            anomaly_type: 异常类型 (e.g., "fighting", "theft", "intrusion", "normal")
            confidence: 异常置信度 (0-1)
        """
        self.anomaly_type_result = {
            "type": anomaly_type,
            "confidence": confidence
        }
        print(f"异常检测结果: {anomaly_type} (置信度: {confidence:.2f})")
        return self.anomaly_type_result
    
    def vlm_final_confirmation(self) -> Dict[str, Any]:
        """
        VLM最终确认
        
        Returns:
            {
                "status": "normal" | "abnormal",
                "alarm_confidence": 0-1,
                "description": "详细描述",
                "reason": "判断原因"
            }
        """
        
        # 构建VLM提示词
        prompt = self._build_vlm_prompt()
        
        print("\nVLM最终确认中...")
        
        try:
            # 调用VLM API - 基于文件路径的方式
            import os
            video_path = os.path.abspath(self.video_path)
            if os.path.exists(video_path):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"video": f"file://{video_path}"}
                        ]
                    }
                ]
            else:
                # 如果没有视频文件，仅基于异常检测结果
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ]
            
            response = dashscope.MultiModalConversation.call(
                model=VIDEO_MODEL,
                messages=messages
            )

            # 判空和结构检查
            if not response or 'output' not in response or not response['output']:
                raise ValueError("VLM API无有效返回")
            choices = response['output'].get('choices')
            if not choices or not isinstance(choices, list) or not choices[0]:
                raise ValueError("VLM API返回choices为空")
            message = choices[0].get('message')
            if not message or 'content' not in message:
                raise ValueError("VLM API返回message无content")
            content = message['content']
            
            # 提取文本内容
            if isinstance(content, list) and len(content) > 0:
                vlm_response = content[0].get('text', str(content))
            else:
                vlm_response = str(content)
            
            self.vlm_result = self._parse_vlm_response(vlm_response)

            print(f"\nVLM确认结果:")
            print(json.dumps(self.vlm_result, ensure_ascii=False, indent=2))

            return self.vlm_result

        except Exception as e:
            print(f"VLM API调用失败: {e}")
            # 降级处理：基于异常检测结果直接判断
            self.vlm_result = self._fallback_decision()
            print(f"\n降级结果:")
            print(json.dumps(self.vlm_result, ensure_ascii=False, indent=2))
            return self.vlm_result
    
    def _build_vlm_prompt(self) -> str:
        """构建VLM提示词（基于UCF-Crime数据集的异常类别）"""
        
        identity_info = json.dumps(self.identity_rules, ensure_ascii=False)
        anomaly_type = self.anomaly_type_result['type']
        confidence = self.anomaly_type_result['confidence']
        
        # 根据异常类型获取判断依据
        criteria = self._get_anomaly_criteria(anomaly_type)
        
        # 仅告知是否超过初筛阈值，不透露具体置信度
        initial_screening = "是" if confidence > 0.5 else "否"
        
        prompt = f"""
你是一个家庭安全监控系统的AI助手。你正在分析来自家庭环境（如客厅、卧室、厨房等）的监控视频。
你需要执行两层检测机制来确保家庭安全：
1. 首先验证深度学习模型给出的犯罪行为检测结果
2. 其次遍历所有犯罪行为类型，逐一思考是否存在其他的犯罪活动

【核心原则】
本系统优先检测真实的安全威胁。仅在有明确证据表明行为不真实时，才判定为非威胁。
- 真实威胁应该基于视频中的实际表现做出判断
- 只有当存在明确的虚假迹象（如玩具枪、明显的表演行为等）时，才降低警报等级
- 不要过度谨慎地将边界案例归类为恶作剧

【监控场景】
家庭环境

【家庭成员信息】
{identity_info}

═══════════════════════════════════════════════════════════════

【第一层：DL模型检测验证】

【DL给出的初步检测结果】
异常类型：{anomaly_type}
初筛结果：{initial_screening}（表示初始检测是否超过初筛阈值）

【该异常类型的判断标准】
{criteria}

请基于视频内容验证DL检测：{anomaly_type} 是否真实存在？
- 检查视频中是否包含符合该异常类型定义的具体行为证据
- 判断这些证据的真实性和强度
- 给出DL检测的验证结果（确认/否认/部分符合）

═══════════════════════════════════════════════════════════════

【第二层：全面犯罪检测（遍历所有犯罪类型）】

现在请逐一检查视频中是否存在以下任何类型的犯罪活动。对于每个犯罪类型，请：
1. 查看视频中是否存在该犯罪行为的迹象，请严格按照后文给出的判断依据
2. 如果存在，描述具体表现和证据
3. 如果不存在，简要说明原因

**逐项检查清单：**

1. **家庭暴力/虐待 (abuse)**
 
2. **人身攻击 (assault)**

3. **打架冲突 (fighting)**

4. **抢劫 (robbery)**
 
5. **盗窃 (stealing)**
  
6. **入室盗窃 (burglary)**
   
7. **枪支伤害 (shooting)**
   
8. **纵火 (arson)**
   
9. **爆炸 (explosion)**
 
10. **财产破坏 (vandalism)**
   
═══════════════════════════════════════════════════════════════

【最终分析和判断】

请基于上述两层检测，给出完整分析（必须用中文回答）：

1. **DL检测验证**：
   - DL初检的 {anomaly_type} 是否真实存在于视频中？
   - 如果存在，请简述证据；如果不存在或仅部分符合，请说明原因

2. **其他犯罪活动检测**：
   - 除了DL检测的 {anomaly_type} 外，是否在上述10种犯罪类型中发现了其他活动？
   - 请列出所有发现的犯罪活动类型和具体表现

3. **综合判断**：
   - 当前视频状态：normal（正常）或 abnormal（存在犯罪活动）
   - 如果存在犯罪活动（DL检测的或独立发现的），请给出报警置信度（0-1）
   - 如果不存在任何犯罪活动，置信度应为 0

4. **详细描述**：
   - 对视频内容的总体描述
   - DL检测的 {anomaly_type} 的验证结果和理由
   - 所有发现的其他犯罪活动的具体表现

5. **推测的行为**：
   - 根据视频特征推测出的所有具体行为和活动

6. **判断原因**：
   - 基于哪些证据和判断标准做出最终决策
   - 为什么需要或不需要报警

请以JSON格式返回结果，格式如下：
{{
    "status": "normal/abnormal",
    "alarm_confidence": 0.X,
    "description": "...",
    "inferred_behavior": "...",
    "dl_validation": "DL检测 {anomaly_type} 的验证结果",
    "other_crimes_detected": "除DL检测外发现的其他犯罪活动列表",
    "reason": "..."
}}

【重要提醒】
- other_crimes_detected 字段：如果没有发现DL检测类型以外的其他犯罪活动，请明确说明"未发现其他犯罪活动"；如果发现了，请列出具体的犯罪类型和表现
- alarm_confidence：只有当存在真实的犯罪活动证据时，才给出大于0的置信度；如果DL检测为normal且独立检测也未发现任何犯罪，置信度应为0
"""
        return prompt.strip()
    
    def _get_anomaly_criteria(self, anomaly_type: str) -> str:
        """获取不同异常类型的判断依据（来自UCF-Crime数据集）"""
        
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
在未获得明确授权的情况下，人员将不属于自己的财产或贵重物品从原有位置取走，
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
  5. 安全状态：所有家庭成员正常活动，无人遭受威胁或伤害"""
        }
        
        return criteria_dict.get(anomaly_type.lower(), "未知异常类型，请基于常见的犯罪特征进行判断")
    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """解析VLM响应"""
        try:
            # 尝试从响应中提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 确保包含必要的字段
                if "dl_validation" not in result:
                    result["dl_validation"] = "未提供"
                if "other_crimes_detected" not in result:
                    result["other_crimes_detected"] = "未检测到其他犯罪活动"
                return result
        except:
            pass
        
        # 如果解析失败，返回默认结果
        return {
            "status": "abnormal",
            "alarm_confidence": 0.5,
            "description": response,
            "inferred_behavior": "无法识别具体行为",
            "dl_validation": "解析失败",
            "other_crimes_detected": "未能进行独立犯罪检测",
            "reason": "基于VLM分析"
        }
    
    def _fallback_decision(self) -> Dict[str, Any]:
        """降级处理：基于异常检测结果直接判断"""
        
        anomaly_type = self.anomaly_type_result['type']
        confidence = self.anomaly_type_result['confidence']
        
        # 不同异常类型对应的报警策略（基于UCF-Crime数据集的13个类别）
        alarm_rules = {
            "abuse": 0.95,              # 虐待极高风险
            "arrest": 0.80,             # 逮捕中等-高风险
            "arson": 0.98,              # 纵火极高风险
            "assault": 0.93,            # 攻击极高风险
            "fighting": 0.95,           # 打架极高风险
            "robbery": 0.95,            # 抢劫极高风险
            "shooting": 0.99,           # 射击极高风险
            "stealing": 0.90,           # 盗窃高风险
            "shoplifting": 0.85,        # 购物盗窃中等-高风险
            "burglary": 0.92,           # 入室盗窃高风险
            "explosion": 0.98,          # 爆炸极高风险
            "vandalism": 0.70,          # 破坏中等风险
            "road_accident": 0.85,      # 交通事故中等-高风险
            "normal": 0.0               # 正常无风险
        }
        
        base_alarm_confidence = alarm_rules.get(anomaly_type.lower(), 0.5)
        final_alarm_confidence = min(base_alarm_confidence * confidence, 1.0)
        
        status = "abnormal" if anomaly_type.lower() != "normal" else "normal"
        
        return {
            "status": status,
            "alarm_confidence": final_alarm_confidence,
            "description": f"异常类型: {anomaly_type}",
            "inferred_behavior": f"检测到的异常行为: {anomaly_type}",
            "reason": "基于异常检测网络结果的降级处理"
        }
    
    def alarm_decision(self, threshold: float = 0.5) -> Tuple[bool, str]:
        """
        根据VLM结果做出最终报警决策
        
        Args:
            threshold: 报警阈值（默认0.5）
            
        Returns:
            (是否报警, 报警原因)
        """
        if not self.vlm_result:
            return False, "未进行VLM确认"
        
        alarm_confidence = self.vlm_result['alarm_confidence']
        should_alarm = alarm_confidence >= threshold
        
        # 检查是否发现了其他犯罪活动（保底机制）
        other_crimes = self.vlm_result.get('other_crimes_detected', '')
        
        # 将other_crimes转换为字符串（处理列表或其他类型）
        if isinstance(other_crimes, list):
            other_crimes_str = '、'.join(other_crimes) if other_crimes else ''
        else:
            other_crimes_str = str(other_crimes) if other_crimes else ''
        
        # 检查是否真的发现了犯罪活动（而不是"未发现"）
        has_other_crimes = (other_crimes_str and 
                           "未检测到" not in other_crimes_str and 
                           "未发现" not in other_crimes_str and 
                           "未能进行" not in other_crimes_str and
                           "无任何" not in other_crimes_str and
                           other_crimes_str.strip() != "")
        
        # 如果发现了其他犯罪活动，必须报警
        if has_other_crimes:
            should_alarm = True
            reason = f"保底检测：发现其他犯罪活动 - {other_crimes_str}"
        else:
            reason = f"报警置信度: {alarm_confidence:.2f} {'≥' if should_alarm else '<'} {threshold}"
        
        print("\n" + "="*60)
        if should_alarm:
            print("【最终决策】 触发报警")
            print("="*60)
            print(f"报警原因: {reason}")
            print(f"事件描述: {self.vlm_result['description']}")
            print(f"推测行为: {self.vlm_result['inferred_behavior']}")
            if 'dl_validation' in self.vlm_result:
                print(f"DL验证结果: {self.vlm_result['dl_validation']}")
            if 'other_crimes_detected' in self.vlm_result:
                print(f"其他犯罪检测: {self.vlm_result['other_crimes_detected']}")
            print(f"判断依据: {self.vlm_result['reason']}")
        else:
            print("【最终决策】 通过检查")
            print("="*60)
            print(f"通过原因: {reason}")
            print(f"事件描述: {self.vlm_result['description']}")
            if 'other_crimes_detected' in self.vlm_result:
                print(f"其他犯罪检测: {self.vlm_result['other_crimes_detected']}")
        print("="*60)
        
        return should_alarm, reason


# ================= 主程序 =================
if __name__ == "__main__":
    import os
    
    print("家庭安全监控系统 - 基于UCF-Crime数据集")
    print("="*60)
    print("系统场景: 家庭环境监控")
    print("检测异常类型:")
    print("  abuse(家庭暴力/虐待), arson(纵火), assault(殴打)")
    print("  fighting(打架), robbery(抢劫), shooting(枪支伤害), stealing(盗窃)")
    print("  vandalism(财产破坏), normal(正常活动)")
    print("="*60)

    # 第一步: 选择检测视频
    print("\n【步骤1】选择检测视频")
    print("-" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_video_dir = os.path.join(base_dir, "test_video")
    test_video_fake_dir = os.path.join(base_dir, "test_video_fake")
    
    video_name = input("请输入视频名称(不需要后缀): ").strip()
    
    # 尝试从test_video文件夹查找
    video_path = os.path.join(test_video_dir, f"{video_name}.mp4")
    if os.path.exists(video_path):
        print(f"在 test_video 文件夹中找到: {video_name}.mp4")
    else:
        # 尝试从test_video_fake文件夹查找
        video_path = os.path.join(test_video_fake_dir, f"{video_name}.mp4")
        if os.path.exists(video_path):
            print(f"在 test_video_fake 文件夹中找到: {video_name}.mp4")
        else:
            print(f"未找到 {video_name}.mp4，使用默认视频 test_video.mp4")
            video_path = "test_video.mp4"

    # 第二步: 输入其他信息
    print("\n【步骤2】输入人物和异常信息")
    print("-" * 60)
    
    try:
        person_count = input("请输入人物数量: ").strip()
        role = input("请输入人物角色: ").strip()
        anomaly_type = input("请输入异常类型 (如 arson, fighting, robbery, stealing 等): ").strip()
        confidence_str = input("请输入异常置信度 (0-1): ").strip()
        confidence = float(confidence_str)
    except Exception as e:
        print(f"输入有误: {e}, 使用默认值")
        person_count = "1"
        role = "unknown"
        anomaly_type = "normal"
        confidence = 1.0

    identity = {
        "person_count": person_count,
        "role": role
    }

    # 第三步: 执行检测
    print("\n【步骤3】执行异常检测")
    print("-" * 60)
    
    pipeline = AnomalyDetectionPipeline()
    pipeline.set_video(video_path)
    pipeline.set_identity_from_rules(identity)
    pipeline.set_anomaly_detection(anomaly_type or "normal", confidence)
    pipeline.vlm_final_confirmation()
    should_alarm, reason = pipeline.alarm_decision(threshold=0.5)
    
    # 最终报警决策输出
    print("\n")
    print("=" * 60)
    if should_alarm:
        print("【系统最终决策】 触发报警")
    else:
        print("【系统最终决策】 无需报警")
    print("=" * 60)
    print(f"决策依据: {reason}\n")
    print(f"决策依据: {reason}\n")
