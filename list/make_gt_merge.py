import numpy as np
import pandas as pd
import re

clip_len = 16

# 合并后的特征列表和标注文件
feature_list = 'cleaned_merged_CLIP_rgbtest.csv'
gt_txt = 'merged_Temporal_Anomaly_Annotation.txt'  # 包含UCF和XD格式的标注

# 读取标注文件
gt_lines = list(open(gt_txt))

# 标签映射，参考combined_train.py中的label_map
label_map = {
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
    'Riot': 'riot'  # 新增的XD标签
}

# 反向标签映射，用于从标签文本获取原始标签
reverse_label_map = {v: k for k, v in label_map.items()}

# 生成连续的异常标签向量
def generate_gt_vector():
    gt = []
    lists = pd.read_csv(feature_list)
    count = 0

    for idx in range(lists.shape[0]):
        name = lists.loc[idx]['path']
        if '__0.npy' not in name:
            continue
        
        fea = np.load(name)
        lens = (fea.shape[0] + 1) * clip_len
        name = name.split('/')[-1]
        name = name[:-7]  # 去除'__0.npy'

        gt_vec = np.zeros(lens).astype(np.float32)
        
        # 判断是否为正常样本
        is_normal = 'Normal' in name or 'label_A' in name
        
        if not is_normal:
            for gt_line in gt_lines:
                # 检查当前特征名是否在标注行中
                if name in gt_line:
                    count += 1
                    
                    # 处理UCF格式 (使用多个空格分隔)
                    if re.match(r'^[A-Z][a-z]*[0-9]+_x264', gt_line.split()[0]):
                        gt_content = re.split(r'\s+', gt_line.strip('\n').split())
                        # UCF格式: [视频名, 标签, 起始1, 结束1, 起始2, 结束2, ...]
                        for i in range(2, len(gt_content), 2):
                            start = int(gt_content[i])
                            end = int(gt_content[i+1])
                            if start != -1 and end != -1:
                                gt_vec[start:end] = 1.0
                    
                    # 处理XD格式 (使用制表符分隔)
                    else:
                        gt_content = gt_line.strip('\n').split('\t')
                        # XD格式: [视频标识, 标签, 起始1, 结束1, 起始2, 结束2, ...]
                        for i in range(2, len(gt_content), 2):
                            start = int(gt_content[i])
                            end = int(gt_content[i+1])
                            if start != -1 and end != -1:
                                gt_vec[start:end] = 1.0
                    
                    break
        
        gt.extend(gt_vec[:-clip_len])

    np.save('gt_merge.npy', gt)
    print(f'Generated gt_merge.npy with {count} anomaly videos')

# 生成异常片段和标签
def generate_gt_segments_labels():
    gt_segment = []
    gt_label = []
    lists = pd.read_csv(feature_list)

    for idx in range(lists.shape[0]):
        name = lists.loc[idx]['path']
        label_text = lists.loc[idx]['label']
        
        if '__0.npy' not in name:
            continue
        
        segment = []
        label = []
        
        # 处理正常样本
        if 'Normal' in label_text or 'label_A' in name:
            fea = np.load(name)
            lens = fea.shape[0] * clip_len
            name = name.split('/')[-1]
            name = name[:-7]  # 去除'__0.npy'
            segment.append([0, lens])
            label.append('Normal')
        
        # 处理异常样本
        else:
            name = name.split('/')[-1]
            name = name[:-7]  # 去除'__0.npy'
            
            for gt_line in gt_lines:
                if name in gt_line:
                    
                    # 处理UCF格式 (使用多个空格分隔)
                    if re.match(r'^[A-Z][a-z]*[0-9]+_x264', gt_line.split()[0]):
                        gt_content = re.split(r'\s+', gt_line.strip('\n').split())
                        # UCF格式: [视频名, 标签, 起始1, 结束1, 起始2, 结束2, ...]
                        anomaly_label = gt_content[1]
                        
                        # 提取所有有效的异常片段
                        for i in range(2, len(gt_content), 2):
                            start = int(gt_content[i])
                            end = int(gt_content[i+1])
                            if start != -1 and end != -1:
                                segment.append([start, end])
                                label.append(anomaly_label)
                    
                    # 处理XD格式 (使用制表符分隔)
                    else:
                        gt_content = gt_line.strip('\n').split('\t')
                        # XD格式: [视频标识, 标签, 起始1, 结束1, 起始2, 结束2, ...]
                        anomaly_label = gt_content[1]
                        
                        # 提取所有有效的异常片段
                        for i in range(2, len(gt_content), 2):
                            start = int(gt_content[i])
                            end = int(gt_content[i+1])
                            if start != -1 and end != -1:
                                segment.append([start, end])
                                label.append(anomaly_label)
                    
                    break
        
        gt_segment.append(segment)
        gt_label.append(label)
    
    np.save('gt_segment_merge.npy', gt_segment)
    np.save('gt_label_merge.npy', gt_label)
    print(f'Generated gt_segment_merge.npy and gt_label_merge.npy with {len(gt_segment)} videos')

if __name__ == '__main__':
    print('Generating gt_merge.npy...')
    generate_gt_vector()
    
    print('\nGenerating gt_segment_merge.npy and gt_label_merge.npy...')
    generate_gt_segments_labels()
    
    print('\nAll files generated successfully!')