import argparse

parser = argparse.ArgumentParser(description='VadCLIP Combined Training')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)  # 使用UCF的参数
parser.add_argument('--attn-window', default=64, type=int)  # 使用XD的参数
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=15, type=int)  # 合并后的类别数：UCF(14) + XD(6) - 重复类别(3) = 17

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='modelcom/combined_model.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='modelcom/combined_checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)  # 使用UCF的参数

# 合并后的训练列表文件路径
parser.add_argument('--combined-train-list', default='list/cleaned_merged_CLIP_rgb.csv', help='Path to merged training list')

# UCF数据集路径
parser.add_argument('--ucf-test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--ucf-gt-path', default='list/gt_ucf.npy')
parser.add_argument('--ucf-gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--ucf-gt-label-path', default='list/gt_label_ucf.npy')

# XD数据集路径
parser.add_argument('--xd-test-list', default='list/cleaned_merged_xd_CLIP_rgbtest.csv')
parser.add_argument('--xd-gt-path', default='list/gt_xd2ucf.npy')
parser.add_argument('--xd-gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--xd-gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=2e-5)  # 使用UCF的参数
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])  # 使用UCF的参数