import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from ucf_test import test as ucf_test
from xd_test import test as xd_test
from utils.dataset import CombinedDataset, UCFDataset, XDDataset
from utils.tools_com import get_prompt_text, get_batch_label
import combined_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

# def train(model, train_loader, ucf_test_loader, xd_test_loader, args, label_map: dict, device):
def train(model, normal_loader, anomaly_loader, ucf_test_loader, xd_test_loader, args, label_map: dict, device):
    model.to(device)

    # 加载UCF的gt数据
    ucf_gt = np.load(args.ucf_gt_path)
    ucf_gtsegments = np.load(args.ucf_gt_segment_path, allow_pickle=True)
    ucf_gtlabels = np.load(args.ucf_gt_label_path, allow_pickle=True)
    
    # 加载XD的gt数据
    xd_gt = np.load(args.xd_gt_path)
    xd_gtsegments = np.load(args.xd_gt_segment_path, allow_pickle=True)
    xd_gtlabels = np.load(args.xd_gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0
    # 后续代码保持不变

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths) 
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 17 * 1e-1  # 调整loss3的权重

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
        
        scheduler.step()
        
        # 在UCF上测试
        print("Testing on UCF dataset...")
        ucf_auc, ucf_ap = ucf_test(model, ucf_test_loader, args.visual_length, prompt_text, ucf_gt, ucf_gtsegments, ucf_gtlabels, device)
        
        # 在XD上测试
        print("Testing on XD dataset...")
        xd_auc, xd_ap, _ = xd_test(model, xd_test_loader, args.visual_length, prompt_text, xd_gt, xd_gtsegments, xd_gtlabels, device)
        
        # 计算平均AP
        avg_ap = (ucf_ap + xd_ap) / 2
        print(f"Average AP: {avg_ap}")
        
        if avg_ap > ap_best:
            ap_best = avg_ap 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)
        
        torch.save(model.state_dict(), 'model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = combined_option.parser.parse_args()
    setup_seed(args.seed)

    # 合并后的label_map，将UCF和XD的标签统一
    label_map = {
        # 所有标签现在都使用统一的格式
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

    # 创建合并后的训练数据集
    normal_dataset = CombinedDataset(args.visual_length, args.combined_train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = CombinedDataset(args.visual_length, args.combined_train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 创建UCF和XD的测试数据集
    ucf_test_dataset = UCFDataset(args.visual_length, args.ucf_test_list, True, label_map)
    ucf_test_loader = DataLoader(ucf_test_dataset, batch_size=1, shuffle=False)
    
    xd_test_dataset = XDDataset(args.visual_length, args.xd_test_list, True, label_map)
    xd_test_loader = DataLoader(xd_test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, ucf_test_loader, xd_test_loader, args, label_map, device)