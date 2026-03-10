import csv
import os

def clean_merged_csv():
    # 输入文件路径
    input_file = '/data1/lihenghao/code/VadCLIP/list/merged_xd_CLIP_rgbtest.csv'
    # 输出文件路径
    output_file = '/data1/lihenghao/code/VadCLIP/list/cleaned_merged_xd_CLIP_rgbtest.csv'
    
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
                label = row[1]
                # 只保留标签中不包含"-"的行
                if "-" not in label:
                    writer.writerow(row)
    
    print(f"处理完成！")
    print(f"输入文件总行数: {sum(1 for line in open(input_file)) - 1}")  # 减去表头行
    print(f"输出文件总行数: {sum(1 for line in open(output_file)) - 1}")  # 减去表头行
    print(f"删除的行数: {sum(1 for line in open(input_file)) - sum(1 for line in open(output_file))}")

if __name__ == '__main__':
    clean_merged_csv()
