#!/usr/bin/env python3

import re

# 定义标签映射
label_mapping = {
    'A': 'Normal',
    'B1-0-0': 'Fighting',
    'B2-0-0': 'Shooting',
    'B4-0-0': 'Riot',
    'B5-0-0': 'Abuse',
    'B6-0-0': 'RoadAccidents',
    'G-0-0': 'Explosion'
}

# 输入文件路径
annotations_file = '/data1/lihenghao/code/VadCLIP/list/annotations.txt'
temporal_file = '/data1/lihenghao/code/VadCLIP/list/Temporal_Anomaly_Annotation.txt'
# 输出文件路径
output_file = '/data1/lihenghao/code/VadCLIP/list/merged_Temporal_Anomaly_Annotation.txt'

# 读取Temporal_Anomaly_Annotation.txt的内容
with open(temporal_file, 'r') as f:
    temporal_content = f.readlines()

# 处理annotations.txt的内容
processed_annotations = []
with open(annotations_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # 分割行内容
        parts = line.split()
        if not parts:
            continue
        
        # 提取视频名称
        video_name = parts[0]
        
        # 提取label信息
        match = re.search(r'_label_([A-Z0-9\-]+)', video_name)
        if not match:
            continue
        
        label = match.group(1)
        
        # 获取对应的异常类型
        anomaly_type = label_mapping.get(label)
        if not anomaly_type:
            continue
        
        # 构建新的行，格式为：视频名称 异常类型 时间点1 时间点2 ...
        new_line = [video_name, anomaly_type]
        
        # 添加时间点，确保是偶数个（每两个为一组）
        timestamps = parts[1:]
        if len(timestamps) % 2 != 0:
            timestamps.append('-1')  # 补全为偶数个
        
        # 确保至少有4个时间点（两组）
        while len(timestamps) < 4:
            timestamps.append('-1')
        
        new_line.extend(timestamps)
        
        # 转换为字符串
        processed_annotations.append('\t'.join(new_line) + '\n')

# 合并内容（先写原Temporal_Anomaly_Annotation.txt的内容，再写处理后的annotations.txt的内容）
with open(output_file, 'w') as f:
    f.writelines(temporal_content)
    f.writelines(processed_annotations)

print(f"合并完成！输出文件为：{output_file}")
print(f"共处理了 {len(processed_annotations)} 行来自 annotations.txt 的数据")
print(f"原 Temporal_Anomaly_Annotation.txt 有 {len(temporal_content)} 行数据")
