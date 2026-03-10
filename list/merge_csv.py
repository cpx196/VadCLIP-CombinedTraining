import csv
import os

def merge_and_modify_csv():
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
    ucf_file = '/data1/lihenghao/code/VadCLIP/list/ucf_CLIP_rgb.csv'
    xd_file = '/data1/lihenghao/code/VadCLIP/list/xd_CLIP_rgb.csv'
    
    # 输出文件路径
    output_file = '/data1/lihenghao/code/VadCLIP/list/merged_CLIP_rgb.csv'
    
    # 打开输出文件
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 处理UCF文件
        with open(ucf_file, 'r') as ucf_infile:
            ucf_reader = csv.reader(ucf_infile)
            # 写入表头
            header = next(ucf_reader)
            writer.writerow(header)
            # 写入UCF数据
            for row in ucf_reader:
                writer.writerow(row)
        
        # 处理XD文件
        with open(xd_file, 'r') as xd_infile:
            xd_reader = csv.reader(xd_infile)
            # 跳过XD文件的表头
            next(xd_reader)
            # 写入XD数据并修改标签
            for row in xd_reader:
                if len(row) >= 2:
                    path = row[0]
                    old_label = row[1]
                    # 修改标签
                    new_label = label_mapping.get(old_label, old_label)  # 如果没有映射则保持原标签
                    writer.writerow([path, new_label])
    
    print(f"文件已成功合并并保存为: {output_file}")

if __name__ == "__main__":
    merge_and_modify_csv()
