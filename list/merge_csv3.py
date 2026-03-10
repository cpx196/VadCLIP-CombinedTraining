import csv

def create_merged_xd_test():
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
    input_file = '/data1/lihenghao/code/VadCLIP/list/xd_CLIP_rgbtest.csv'
    
    # 输出文件路径
    output_file = '/data1/lihenghao/code/VadCLIP/list/merged_xd_CLIP_rgbtest.csv'
    
    # 打开输入文件和输出文件
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 写入表头
        header = next(reader)
        writer.writerow(header)
        
        # 处理数据行
        for row in reader:
            if len(row) >= 2:
                path = row[0]
                old_label = row[1]
                # 修改标签
                new_label = label_mapping.get(old_label, old_label)  # 如果没有映射则保持原标签
                writer.writerow([path, new_label])
    
    print(f"文件已成功创建并保存为: {output_file}")
    print(f"输入文件总行数: {sum(1 for line in open(input_file)) - 1}")  # 减去表头行
    print(f"输出文件总行数: {sum(1 for line in open(output_file)) - 1}")  # 减去表头行

if __name__ == "__main__":
    create_merged_xd_test()
