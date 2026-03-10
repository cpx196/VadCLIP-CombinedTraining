import csv
import os

def replace_paths_in_csv(input_file, output_file):
    """
    替换CSV文件中的路径
    """
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 写入表头
        header = next(reader)
        writer.writerow(header)
        
        # 处理每一行数据
        for row in reader:
            if len(row) >= 2:
                # 替换路径
                old_path = row[0]
                # 将/home/xbgydx/Desktop/UCFClipFeatures/替换为/data1/lihenghao/data/
                new_path = old_path.replace('/home/xbgydx/Desktop/', '/data1/lihenghao/data/')
                row[0] = new_path
                writer.writerow(row)

if __name__ == "__main__":
    input_csv = "/data1/lihenghao/code/VadCLIP/list/xd_CLIP_rgbtest.csv"
    output_csv = "/data1/lihenghao/code/VadCLIP/list/xd_CLIP_rgbtest1.csv"
    
    replace_paths_in_csv(input_csv, output_csv)
    print(f"文件已成功转换并保存为: {output_csv}")
