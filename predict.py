

import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
# from models import CompletionNetwork
from model.swin_transformer import swin_tiny_patch4_window7_224 as create_model
# from model.swin_transformer import swin_base_patch4_window7_224 as create_model
# from model.resnet import resnet50 as create_model
from utils import *
import copy
# from kUtil import *
import pandas as pd

# setup_seed(42)

np.set_printoptions(threshold=np.inf)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='UJI')  # or 'CNG'
# parser.add_argument('--config', type=str, default='/nfs/UJI_LIB/data/updateResult/floor3/swinT_mask90/config.json')
# parser.add_argument('--config', type=str, default='/nfs/UJI_LIB/data/updateResult/')
parser.add_argument('--config', type=str, default='/nfs/UJI_LIB/data/updateResult/')
# parser.add_argument('--config', type=str, default='/nfs/UJI_LIB/data/swin_base_patch4_window7_224/')
# parser.add_argument('--config', type=str, default='/nfs/UJI_LIB/data/updateResult_random/')
# parser.add_argument('--test_dir', type=str, default='/nfs/UJI_LIB/data/updateDataset/floor_3/')
parser.add_argument('--test_dir', type=str, default='/nfs/UJI_LIB/data/updateDataset/')
# parser.add_argument('--modelDir', type=str, default='/nfs/UJI_LIB/data/updateResult/floor3/swinT_mask90/')
parser.add_argument('--modelDir', type=str, default='/nfs/UJI_LIB/data/updateResult/')
# parser.add_argument('--modelDir', type=str, default='/nfs/UJI_LIB/data/swin_base_patch4_window7_224/')
# parser.add_argument('--modelDir', type=str, default='/nfs/UJI_LIB/data/updateResult_random/')
# parser.add_argument('output_img')
# parser.add_argument('--output_imgDir', type=str, default='/nfs/UJI_LIB/data/updateResult/floor3/swinT_mask90/')
# parser.add_argument('--output_imgDir', type=str, default='/nfs/UJI_LIB/data/updateResult_lucky/')
# parser.add_argument('--output_imgDir', type=str, default='/nfs/UJI_LIB/data/mask_test/')
parser.add_argument('--output_imgDir', type=str, default='/nfs/UJI_LIB/data/updateResult/')
# parser.add_argument('--output_imgDir', type=str, default='/nfs/UJI_LIB/data/swin_base_patch4_window7_224/')
# parser.add_argument('--output_imgDir', type=str, default='/nfs/radiomap_results/swinT_mask35/add_possion/')
# parser.add_argument('--model', type=str, default='./weights/model_w7_d3_h6_l1_w_m75.pth')
parser.add_argument('--mask_ratio', type=float, default=0.9)
parser.add_argument('--floor', type=int, default=3)  # 3 | 5
parser.add_argument('--month', type=int, default=11)  # range[1, 25]
# parser.add_argument('--method', type=str, default='resnet50')
parser.add_argument('--method', type=str, default='swinT')
parser.add_argument('--seed', type=int, default=0)

setup_seed(parser.parse_args().seed)

model_name = "swin"  # "swin"

def main(args):

    # args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.test_dir = os.path.expanduser(args.test_dir)
    args.modelDir = os.path.expanduser(args.modelDir)
    args.output_imgDir = os.path.expanduser(args.output_imgDir)
    mask_ratio = args.mask_ratio

    if args.dataset == 'UJI':
        modelDir = oj(args.modelDir, f'floor{args.floor}', f'{args.method}_mask{int(args.mask_ratio*100)}')
        test_dir = oj(args.test_dir, f'floor_{args.floor}', 'predict', f'month_{args.month}')
        config_dir = oj(args.config, f'floor{args.floor}', f'{args.method}_mask{int(args.mask_ratio*100)}', 'config.json')
        output_imgDir = oj(args.output_imgDir, f'floor{args.floor}', f'month{args.month}',
                        f'{args.method}_mask{int(args.mask_ratio*100)}')

    elif args.dataset == 'CNG':
        modelDir = args.modelDir
        test_dir = oj(args.test_dir, 'predict')
        config_dir = oj(args.config, 'config.json')
        output_imgDir = oj(args.output_imgDir)

    # =============================================
    # Load model
    # =============================================
    with open(config_dir, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    # model = CompletionNetwork()

    num_classes = 3 * 8 * 11
    model = create_model(num_classes=num_classes)

    # model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.load_state_dict(torch.load(oj(modelDir, f"model_{model_name}_m{int(args.mask_ratio*100)}.pth"), map_location='cpu'))

    # =============================================
    # Predict
    # =============================================
    err_ls = []
    mae_ls = []
    img_dt_err = {}
    predictJson = {}
    floorTrue_df = pd.DataFrame(columns=['x', 'y', 'floor'])
    floorInpa_df = pd.DataFrame(columns=['x', 'y', 'floor'])

    # test_dir = args.test_dir
    test_set = ls(test_dir)

    img0 = Image.open(oj(test_dir, test_set[0]))
    img0 = transforms.ToTensor()(img0)
    img0 = torch.unsqueeze(img0, dim=0)

    mask = gen_input_mask_random(
        shape=(1, 1, img0.shape[2], img0.shape[3]),
        mask_ratio=mask_ratio
    )
    # mask = gen_bad_mask_random(
    #     shape=(1, 1, img0.shape[2], img0.shape[3]),
    #     mask_ratio=mask_ratio
    # )

    wd_maskRp(mask, args.floor, oj(output_imgDir, 'mask_data'))

    for input_img in test_set:

        img = Image.open(oj(test_dir, input_img))
        # convert img to tensor
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0)

        # inpaint
        model.eval()
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask
            # input = torch.cat((x_mask, mask), dim=1)
            inpainted = model(x_mask)
            # inpainted replaced unmask
            inpainted = x + (inpainted - x) * mask  # mask: 1, unmask:0
            # inpainted = poisson_blend(x_mask, inpainted, mask)

            # x, x_mask, inpainted
            # x_origion =
            x_origion = transforms.ToPILImage()(x[0]).convert('L')
            x_mask = transforms.ToPILImage()(x_mask[0]).convert('L')
            inpainted = transforms.ToPILImage()(inpainted[0]).convert('L')


            x_origion_array = np.array(x_origion, dtype=np.float64)
            x_mask = np.array(x_mask, dtype=np.float64)
            inpainted = np.array(inpainted, dtype=np.float64)

            # print(x_origion_array)
            x_origion_array = toRSSIMap(x_origion_array)
            x_mask = toRSSIMap(x_mask)
            inpainted = toRSSIMap(inpainted)

            # plotSns
            index = int(input_img.split('.')[0])
            origion_dir = oj(output_imgDir, 'origion_offaxis')
            # origion_dir = oj(args.output_imgDir, 'origion_tourf')
            if not os.path.exists(origion_dir):
                os.makedirs(origion_dir)
            plotSns(index, x_origion_array, origion_dir)
            # plotContourf(index, x_origion_array, origion_dir)


            mask_dir = oj(output_imgDir, 'mask_offaxis')
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            plotSns(index, x_mask, mask_dir)

            inpaint_dir = oj(output_imgDir, 'inpaint_offaxis')
            # inpaint_dir = oj(args.output_imgDir, 'inpaint_tourf')
            if not os.path.exists(inpaint_dir):
                os.makedirs(inpaint_dir)
            plotSns(index, inpainted, inpaint_dir)
            # plotContourf(index, inpainted, inpaint_dir)

            x_origion_ls = x_origion_array.flatten()
            inpainted_ls = inpainted.flatten()

            floorTrue_df[f'{index}'] = x_origion_ls
            floorInpa_df[f'{index}'] = inpainted_ls

            error = sum(((inpainted_ls - x_origion_ls) ** 2) ** 0.5) / (len(x_origion_ls))
            err_ls.append(error)
            img_dt_err[input_img] = error
            mae = np.linalg.norm(x_origion_ls - inpainted_ls, ord=1) / (len(x_origion_ls))
            mae_ls.append(mae)

        print(f'{input_img} was done.')
    mean_err = np.mean(err_ls)
    mean_mae = np.mean(mae_ls)
    predictJson['mean_err'] = mean_err
    predictJson['mean_mae'] = mean_mae
    predictJson['img_dt_err'] = img_dt_err

    err_dir = oj(output_imgDir, 'errData2')
    if not os.path.exists(err_dir):
        os.makedirs(err_dir)

    CDF(err_ls, err_dir)
    with open(oj(err_dir, 'err_result.json'), 'w') as f:
        json.dump(predictJson, f, indent=4)
    saveTrInpa(floorTrue_df, floorInpa_df, args.floor, oj(output_imgDir, 'inpainted_data2'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
