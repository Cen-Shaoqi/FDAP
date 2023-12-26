import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# from radiomap2 import T_MAP, V_MAP, gen_mask, cal_dists, get_AP_location
# from model import swin_ae_patch2_window3_48 as create_model
# from model.swin_transformer import swin_tiny_patch4_window7_224 as create_model
# from model.swin_transformer import swin_small_patch4_window7_224 as create_model
# from model.swin_transformer import swin_base_patch4_window7_224 as create_model
from model.resnet import resnet50 as create_model
# from model.vgg import vgg as create_model
# from model.cswin import CSWin_64_12211_tiny_224 as create_model
from utils import *
from datasets import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

model_name = "swin"  # "swin" "resnet50"

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    if args.dataset == 'UJI':
        # result_dir = oj(args.result_dir, f'floor{args.floor}', f'month{args.month}', f'{args.method}_mask{int(args.mask_ratio*100)}')
        result_dir = oj(args.result_dir, f'floor{args.floor}', f'{args.method}_mask{int(args.mask_ratio*100)}')
        data_dir = oj(args.data_dir, f'floor_{args.floor}')
    elif args.dataset == 'CNG':
        result_dir = args.result_dir
        data_dir = args.data_dir

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    tb_writer = SummaryWriter()


    trnsfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(
        os.path.join(data_dir, 'train'),
        trnsfm,
        recursive_search=args.recursive_search)
    test_dataset = ImageDataset(
        os.path.join(data_dir, 'test'),
        trnsfm,
        recursive_search=args.recursive_search)
    val_dataset = ImageDataset(
        os.path.join(data_dir, 'val'),
        trnsfm,
        recursive_search=args.recursive_search)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    if args.mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(
            total=len(train_dataset.imgpaths),
            desc='computing mean pixel value of training dataset...')
        for imgpath in train_dataset.imgpaths:
            img = Image.open(imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(train_dataset.imgpaths)
        pbar.close()
    else:
        mpv = np.array(args.mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))
    args_dict = vars(args)
    args_dict['mpv'] = mpv_json
    with open(os.path.join(
            result_dir, 'config.json'),
            mode='w') as f:
        json.dump(args_dict, f)

    # make mpv & alpha tensors
    mpv = torch.tensor(
        mpv.reshape(1, 3, 1, 1),
        dtype=torch.float32).to(device)
    num_classes = 3 * 8 * 11
    model = create_model(num_classes=num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    val_err = 2000
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                mask_ratio=args.mask_ratio,
                                                mpv=mpv)

        # validate
        if epoch > 1000 or epoch % 30 == 0:
            val_mse_err = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         mpv=mpv)


            test_mse_err = 0
            tags = ["train_loss", "val_mse_loss", "val_mae_loss", "learning_rate", "test_mse_err", "test_mae_err"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], val_mse_err, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

            if val_mse_err < val_err:
                val_err = val_mse_err
                save_name = oj(result_dir, f"model_{model_name}_m{int(args.mask_ratio*100)}.pth")
                torch.save(model.state_dict(), save_name)
                test_mse_err = evaluate(model=model,
                                                      data_loader=test_loader,
                                                      device=device,
                                                      epoch=epoch,
                                                      mpv=mpv)
                print(save_name)
                # print("v_mse: %.6f, v_mae: %.6f, t_mse: %.6f, t_mae: %.6f, lr:%f" % (val_mse_err, val_mae_err, test_mse_err, test_mae_err, optimizer.param_groups[0]["lr"]))
                print("v_mse: %.6f, t_mse: %.6f, lr:%f" % (val_mse_err, test_mse_err, optimizer.param_groups[0]["lr"]))
            tb_writer.add_scalar(tags[4], test_mse_err, epoch)
            # tb_writer.add_scalar(tags[5], test_mae_err, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='UJI')  # or 'CNG'
    parser.add_argument('--result_dir', type=str, default='/nfs/UJI_LIB/data/updateResult/floor3/swinT_mask90/')
    parser.add_argument('--result_dir', type=str, default='/nfs/UJI_LIB/data/updateResult/')
    parser.add_argument('--mask_ratio', type=float, default=0.9)
    # parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--recursive_search', action='store_true', default=False)
    parser.add_argument('--mpv', nargs=3, type=float, default=None)
    parser.add_argument('--floor', type=int, default=5)  # or 5
    parser.add_argument('--month', type=int, default=1)  # range[1, 25]
    parser.add_argument('--method', type=str, default='swinT')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--scene', default='SEIT', type=str)

    opt = parser.parse_args()

    main(opt)
