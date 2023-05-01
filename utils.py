import os
import sys
import json
import pickle
import random
import cv2
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
from torch.nn.functional import mse_loss
import seaborn as sns
import statsmodels.api as sm
import torchvision.transforms as transforms
from torch.backends import cudnn

# np.set_printoptions(threshold=np.inf)
x_ls = [3.13, 4.13, 5.13, 6.33, 7.52, 8.52, 9.52, 10.72, 11.91, 12.91, 13.91]
y_ls = [16.70, 18.49, 20.28, 22.06, 23.85, 25.64, 27.43, 29.22]


def oj(*args):
    return os.path.join(*args)


def ls(path):
    return os.listdir(path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(42)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)


def gen_input_mask_random(
        shape, mask_ratio=0.5, mask_area=None):
    random.seed(42)  # 保证随机结果可复现
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    token_count = mask_h * mask_w
    mask_count = int(np.ceil(token_count * mask_ratio))
    for i in range(bsize):
        if mask_area is not None:
            # print("mask_area is not None.")
            mask[i, :] = mask[i, :] + mask_area
        else:
            mask_idx = np.random.permutation(token_count)[:mask_count]
            single_mask = np.zeros(token_count)
            single_mask[mask_idx] = 1

            single_mask = single_mask.reshape((mask_h, mask_w))

            mask[i, :] = mask[i, :] + single_mask

    return mask


def gen_bad_mask_random(
        shape, mask_ratio=0.5, mask_area=None):
    random.seed(42)  # 保证随机结果可复现
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    token_count = mask_h * mask_w
    sample_count = int(np.ceil(token_count * (1 - mask_ratio))) - 1
    for i in range(bsize):
        if mask_area is not None:
            # print("mask_area is not None.")
            mask[i, :] = mask[i, :] + mask_area
        else:
            # mask_idx = np.random.permutation(token_count)[:mask_count]
            # single_mask = np.zeros(token_count)
            front_mask_idx = np.random.permutation(int(token_count/4))[:sample_count]
            front_mask = np.ones(int(token_count/4))
            front_mask[front_mask_idx] = 0
            back_mask = np.ones(token_count - int(token_count/4))
            single_mask = np.concatenate((back_mask, front_mask))
            single_mask = single_mask.reshape((mask_h, mask_w))

            mask[i, :] = mask[i, :] + single_mask

    return mask

def gen_mask_area_random(mask_ratio, mask_size):
    random.seed(42)  # 保证随机结果可复现
    mask_w, mask_h = mask_size
    token_count = mask_h * mask_w
    mask_count = int(np.ceil(token_count * mask_ratio))
    mask_idx = np.random.permutation(token_count)[:mask_count]
    single_mask = np.zeros(token_count)
    single_mask[mask_idx] = 1

    single_mask = single_mask.reshape((mask_h, mask_w))
    return single_mask


def train_one_epoch(model, optimizer, data_loader, device, epoch, mask_ratio=0.5, mpv=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    # for step, data in enumerate(data_loader):
    step_temp = 0
    for step, images in enumerate(data_loader):
        images = images.to(device)
        # print(data)
        # images, labels = data

        # change start ...
        mask = gen_input_mask_random(
            shape=(images.shape[0], 1, images.shape[2], images.shape[3]),
            mask_ratio=mask_ratio,
            mask_area=gen_mask_area_random(
                mask_ratio=mask_ratio,
                mask_size=(images.shape[3], images.shape[2])
            )
        ).to(device)
        # print("images shape: ", images.shape)
        # print("mask shape: ", mask.shape)
        x_mask = images - images * mask + mpv * mask
        # print("x_mask shape: ", x_mask.shape)
        # print(x_mask)
        # images = torch.cat((x_mask, mask), dim=1)
        # change end ...

        sample_num += x_mask.shape[0]
        # print(x_mask)
        # print(x_mask.shape)
        # print("------------ images -------------------")
        # print(images)
        pred = model(x_mask.to(device))
        # print("------------ pred -------------------")
        # print(pred)
        # print("x_mask shape: ", x_mask.shape)
        # print("pred shape: ", pred.shape)
        # print(images.shape)
        # print(pred.shape)
        # print(mask.shape)
        loss = completion_network_loss(images, pred, mask)
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        #
        # loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        # data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        step_temp = step

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    return accu_loss.item() / (step_temp + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, mask_ratio=0.5, mpv=None):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    step_temp = 0
    # for step, data in enumerate(data_loader):
    for step, images in enumerate(data_loader):
        # images, labels = data
        images = images.to(device)
        sample_num += images.shape[0]

        mask = gen_input_mask_random(
            shape=(images.shape[0], 1, images.shape[2], images.shape[3]),
            mask_ratio=mask_ratio,
            mask_area=gen_mask_area_random(
                mask_ratio=mask_ratio,
                mask_size=(images.shape[3], images.shape[2])
            )
        ).to(device)

        x_mask = images - images * mask + mpv * mask

        # print(x_mask.shape)
        # print(images)
        pred = model(x_mask.to(device))
        # print(pred)
        # pred_classes = torch.max(pred, dim=1)[1]
        loss = completion_network_loss(images, pred, mask)
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)
        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
        step_temp = step

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    return accu_loss.item() / (step_temp + 1)


def plotSns(img_id, seabornData_np, plotSaveDir):
    sns.set()
    plotData = seabornData_np
    f, ax = plt.subplots(figsize=(9, 6))
    # sns.heatmap(plotData, cmap='YlGnBu', ax=ax, vmin=-100, vmax=-30, annot=True)
    sns.heatmap(plotData, cmap='YlGnBu', ax=ax, vmin=-100, vmax=-30)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=90, horizontalalignment='right')
    plt.axis('off')
    # plt.show()
    plt.savefig(oj(plotSaveDir, f'{img_id}.eps'), format="eps", dpi=300)
    plt.close()


def plotContourf(img_id, seabornData_np, plotSaveDir):
    sns.set()
    plotData = seabornData_np
    f, ax = plt.subplots(figsize=(9, 6))
    plt.contourf(plotData)
    # sns.heatmap(plotData, cmap='YlGnBu', ax=ax, vmin=-100, vmax=-30)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=90, horizontalalignment='right')
    plt.savefig(oj(plotSaveDir, f'{img_id}.png'))
    plt.close()


def wd_maskRp(mask, floor, saveDir):

    # input mask: tensor.shape() = 4
    mask = mask.numpy()[0][0]

    exrp_idx = np.where(mask == 1)
    unrp_idx = np.where(mask == 0)

    unrpCoord_ls = []
    for i in range(len(unrp_idx[0])):
        unrpCoord_ls.append([x_ls[unrp_idx[1][i]], y_ls[unrp_idx[0][i]], floor])

    exrpCoord_ls = []
    for i in range(len(exrp_idx[0])):
        exrpCoord_ls.append([x_ls[exrp_idx[1][i]], y_ls[exrp_idx[0][i]], floor])

    unrpCoord_np = np.array(unrpCoord_ls)
    exrpCoord_np = np.array(exrpCoord_ls)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    np.savetxt(oj(saveDir, "unrpCoord.csv"), unrpCoord_np, delimiter=",", fmt='%.02f')
    np.savetxt(oj(saveDir, "exrpCoord.csv"), exrpCoord_np, delimiter=",", fmt='%.02f')


def saveTrInpa(floorTrue_df, floorInpa_df, floor, saveDir):
    xdf_ls = []
    ydf_ls = []
    floor_ls = []
    for y in y_ls:
        for x in x_ls:
            xdf_ls.append(x)
            ydf_ls.append(y)
            floor_ls.append(floor)
    floorTrue_df['x'] = xdf_ls
    floorTrue_df['y'] = ydf_ls
    floorTrue_df['floor'] = floor_ls

    floorInpa_df['x'] = xdf_ls
    floorInpa_df['y'] = ydf_ls
    floorInpa_df['floor'] = floor_ls

    # save dataframe to saveDir
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    floorTrue_df.to_csv(oj(saveDir, 'true.csv'), header=True, index=False)
    floorInpa_df.to_csv(oj(saveDir, 'inpainted.csv'), header=True, index=False)


def toRSSIMap(input_np):
    # input: numpy
    # rssi_np = ((input_np - 53) * 100 / 200) - 101

    return ((input_np - 53) * 100 / 200) - 101


def PILtoNumpy(input_PIL):
    # input: PIL
    # output: numpy
    return np.array(input_PIL, dtype=np.float64)


def getCNGtemplate():
    # 1: can arrive
    # 0: cannot arrive
    CNGtemplate = np.ones((32, 55))
    CNGtemplate[1: 13] = 0  # 0
    CNGtemplate[18: 31] = 0  # 0
    CNGtemplate[:, :2] = 1  # 1
    CNGtemplate[:, 50:] = 1  # 1
    return CNGtemplate


def CDF(err_ls, saveDir):
    ecdf = sm.distributions.ECDF(err_ls)
    x = np.linspace(min(err_ls), max(err_ls))
    y = ecdf(x)
    plt.step(x, y)
    plt.title('CDF')
    plt.xlabel("Reconstruction Accuracy (dBm)")
    plt.ylabel('CDF')
    plt.savefig(oj(saveDir, 'err_cdf.png'))
    plt.close()


def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret