import numpy as np
# from point_detect.point_quality import get_D, get_C, get_S, get_error
from PIL import Image
from utils import *
from scipy import ndimage
import pandas as pd
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
# from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import scipy

np.set_printoptions(threshold=np.inf)

mask_ratio = 0.9
floor = 3
old_month = 1
month = 12
badp_threshold = -90
goodp_threshold = -70
# setup_seed(seed)

test_dir = oj('/nfs/UJI_LIB/data/updateDataset/', f'floor_{floor}', 'predict', f'month_{month}')
test_set = ls(test_dir)


def sliding_window(data, w_size):
    a, b = data.shape
    m, n = w_size
    result = []
    for i in range(0, a, m):
        line = []
        for j in range(0, b, n):
            x = data[i:i+m, j:j+n]  # 选取区域
            # quality_detect algorithm
            # if satified:
            # update algorithm
            # else:
            #
            line.append(np.sum(x)/(n*m))
        result.append(line)
    return np.array(result)


def sliding_window2(data, w_size, stride=1):
    a, b = data.shape
    m, n = w_size
    result = []
    for i in range(0, a, stride):
        line = []
        for j in range(0, b, stride):
            x = data[i:i+m, j:j+n]  # 选取区域
            line.append(np.sum(x)/(n*m))
        result.append(line)
    return np.array(result)


def local_update(coord_ls, target, source):
    for coord in coord_ls:
        target = target.append(source.loc[(source['x'] == coord[0]) & (source['y'] == coord[1])])

    return target


def deal_tuple(list_tuple, start_x, start_y):
    x = [i[0] for i in list_tuple]
    y = [i[1] for i in list_tuple]

    x = [j + start_x for j in x]
    y = [j + start_y for j in y]
    return list(zip(x, y))


def cal_quality(data, start_x, start_y):
    # judge quality is Great?
    isGreat = False
    update_coord_ls = None
    observe_array = np.count_nonzero(data)
    observe_sum = np.sum(observe_array)
    score = observe_sum/np.size(data)
    # if score >= 0.5:
    if score >= 0.3:
        isGreat = True

    # if great, return the local signals' index
    if isGreat and (np.size(data) != 1):
        matrix_index = np.where(data == 0)
        update_coord_ls = deal_tuple(list(zip(matrix_index[0], matrix_index[1])), start_x, start_y)
    return isGreat, update_coord_ls


def map_coord(coord_ls):
    map_coord_ls = []
    for coord in coord_ls:
        map_coord_ls.append([x_ls[coord[1]], y_ls[coord[0]], floor])
    return map_coord_ls


# def get_outdateCoord(data_um, data_outdate):
#     coord_ls = []
#     exist = data_um.iloc[:,:3]
#     full = data_outdate.iloc[:,:3]
#     for
#     return coord_ls


def get_wSize_ls(h, w):
    wSize_ls = []
    if (w % 2) != 0:
        w = w+1
    for i in [1, 2, 4]:
        if (int(h/i) <= 1) or (int(w/i) <= 1):
            continue
        w_size = [int(h/i), int(w/i)]
        wSize_ls.append(w_size)
    return wSize_ls


def get_localArea(sampled_coord, map_m, w_size, stride=1):
    h, w = map_m.shape
    wSize_ls = get_wSize_ls(h, w)
    for w_size in wSize_ls:
        m, n = w_size[0], w_size[1]
        update_ls = []
        for i in range(0, h, stride):
            # line = []
            for j in range(0, w, stride):
                local_m = map_m[i:i + m, j:j + n]
                # get quality
                isGreat, update_coord_ls = cal_quality(local_m, i, j)
                # get local update singals' idx
                if isGreat and (update_coord_ls != None):
                    update_ls += update_coord_ls

    update_ls = list(set(update_ls))
    localUpdate_coord = map_coord(update_ls)
    # 和采样RP合并
    localUpdate_coord = sampled_coord.values.tolist() + localUpdate_coord
    return localUpdate_coord


def divide_localArea(data, localArea):

    train_df = pd.DataFrame(columns=['x', 'y'])
    for coord in localArea:
        train_df = train_df.append(data.loc[(data['x'] == coord[0]) & (data['y'] == coord[1])])

    data2 = data.merge(train_df, how='left', indicator=True)
    test_df = data2[data2._merge == 'left_only'].copy()
    test_df.drop('_merge', axis=1, inplace=True)
    return train_df, test_df


def plot_scatter(coords1, coords2):
    plt.scatter(coords1[:, 0], coords1[:, 1], color='#1597A5', label='Realiable Signals', marker='o')
    plt.scatter(coords2[:, 0], coords2[:, 1], color='#FEB3AE', label='Sampled Signals', marker='x')

    # plt.legend(loc='upper right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=5, frameon=False)
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.savefig(oj(savPath, f'RP_scatter.eps'), format="eps")
    plt.show()
    plt.close()


def sliding_window_renew(sampled_coord, data_um, data_update, data_outdate, map_m, w_size, stride=1):
    col_num = data_um.shape[1]
    # get local area
    localUpdate_coord = get_localArea(sampled_coord, map_m, w_size, stride)
    # plot_scatter(np.array(localUpdate_coord), np.array(sampled_coord))

    # divide local and not in local area
    # 选取除前三列的值，并转为 np.array
    T0_train, T0_test = divide_localArea(data_outdate, localUpdate_coord)
    T1_train, T1_test = divide_localArea(data_update, localUpdate_coord)
    X_train = np.array(T0_train.iloc[:, 3:])
    y_train = np.array(T1_train.iloc[:, 3:])
    X_test = np.array(T0_test.iloc[:, 3:])

    pred_header_ls = T1_train.columns.values
    coords = np.array(T0_test.iloc[:, :3])
    # 回归模型，参数
    pls_model_setup = PLSRegression(scale=True)
    param_grid = {'n_components': range(1, 4)}

    # GridSearchCV 自动调参
    gsearch = GridSearchCV(pls_model_setup, param_grid)

    # 在训练集上训练模型
    pls_model = gsearch.fit(X_train, y_train)

    # 预测
    T1_pred = pls_model.predict(X_test)
    T1_pred = np.concatenate((coords, T1_pred), axis=1)
    T1_pred = pd.DataFrame(T1_pred, columns=pred_header_ls)

    # local update
    # data_um = local_update(localUpdate_coord, data_um, data_update)

    data = T1_pred
    data = data.append(T1_train)
    data.sort_values(by=['y', 'x'], ascending=True, inplace=True)
    data[data <= -100] = -105

    # return data.iloc[:, :col_num]
    return data


def get_mask(seed):
    # get mask
    img0 = Image.open(oj(test_dir, test_set[0]))
    img0 = transforms.ToTensor()(img0)
    img0 = torch.unsqueeze(img0, dim=0)

    if seed < 1000:
        mask = gen_input_mask_random(
                shape=(1, 1, img0.shape[2], img0.shape[3]),
                mask_ratio=mask_ratio
        )
    else:
        mask = gen_bad_mask_random(
                shape=(1, 1, img0.shape[2], img0.shape[3]),
                mask_ratio=mask_ratio
        )

    # wd_maskRp(mask, floor, "/nfs/UJI_LIB/data/point_detect/points/")
    mask = np.array(mask, dtype=np.int32)

    mask[mask==1] = 2
    mask[mask==0] = 1
    mask[mask==2] = 0
    return mask[0][0]


def calculate_error(predict, true):
    err_ls = []
    AP_list = true.columns.values.tolist()[3:]

    # calculate each col's (AP's) error
    for AP in AP_list:
        predict_AP = predict.loc[:, [AP]]
        true_AP = true.loc[:, [AP]]
        mae = mean_absolute_error(predict_AP, true_AP)
        err_ls.append(mae)

    return err_ls


def plot_cdf(data1, data2):
    ecdf1 = sm.distributions.ECDF(data1)
    ecdf2 = sm.distributions.ECDF(data2)

    x = np.linspace(min(data1 + data2), max(data1 + data2))
    y1 = ecdf1(x)
    y2 = ecdf2(x)
    plt.step(x, y1, label='(w/) Local Constraints', color='#F66F69', linewidth=2)
    plt.step(x, y2, label='(w/o) Local Constraints', color='#1597A5', linewidth=2)
    plt.xlabel("Reconstruction Error (dBm)", {"size": 15})
    plt.ylabel("CDF", {"size": 15})
    plt.legend()

    # plt.savefig(oj('../plt_data/', f'local_constraints.eps'), format="eps")
    plt.show()
    plt.close()


def filterBetter(local_data, global_data, rule="max"):
    better, worse = [], []
    for i in range(len(local_data)):
        # if local_data[i] < global_data[i]:
        if ((global_data[i] - local_data[i]) >= 0.45) and (global_data[i] >= 3.0):
        # if (local_data[i] < global_data[i]) and (global_data[i] >= 4.0):
            # if rule == "max":
            #     # continue
            # else:
            better.append(local_data[i])
            worse.append(global_data[i])
    return better, worse

def plot_err_bar(local_data, global_data, savePath):
    showNum = 15
    # filter the value when local < global
    better, worse = filterBetter(local_data, global_data, rule="max")
    better = better[:showNum]
    worse = worse[:showNum]
    # plot
    fig, axes = plt.subplots()
    x = list(range(0, showNum))
    plt.scatter(x, worse, color='#587498', label='Before Correction', marker='x', zorder=10)
    plt.scatter(x, better, color='#e86850', label='After Correction', marker='o', zorder=10)
    axes.set_xticks(np.linspace(0, len(x), num=6))
    axes.set_yticks(np.linspace(0, 10, num=6))
    axes.set_xticks(np.linspace(1, len(x), num=showNum), minor=True)
    axes.set_yticks(np.linspace(1, 10, num=10), minor=True)
    plt.legend(loc="upper right")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            ncol=5, frameon=False)
    plt.grid(linestyle='--', which="major", alpha=0.6, c='k', zorder=0)
    plt.grid(linestyle='--', which="minor", alpha=0.3, c='k', zorder=0)
    plt.xlabel("Index of Radio Maps", {"size": 15})
    plt.ylabel("Reconstruction Error (dB)", {"size": 15})
    plt.savefig(oj(savePath, f'err_bar.eps'), format="eps", dpi=600)
    plt.show()
    plt.close()


def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = list(set1.intersection(set2))
    iset.sort(key=list(list1).index)
    return iset


# def merge_outdateTrue_AP(data_outdate_header_ls, data_true_header_ls):
#
#     return data_outdate_merge, data_true_merge

def main():
    # set seed
    seed = 7
    setup_seed(seed)
    # read data
    dataDir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask_r{seed}', f'floor{floor}', f'month{month}',
                         f'swinT_mask{int(mask_ratio * 100)}')

    unmaskDir = oj(dataDir, 'mask_data', 'unmaskRss.csv')
    data_um = pd.read_csv(unmaskDir)

    sampled_coord_Dir = oj(dataDir, 'mask_data', 'unrpCoord.csv')
    sampled_coord = pd.read_csv(sampled_coord_Dir, header=None)

    data_update_Dir = oj(dataDir, 'inpainted_data', 'inpainted.csv')
    data_update = pd.read_csv(data_update_Dir)
    data_update_header_ls = data_update.columns.values

    # print(data_update)
    data_true_Dir = oj(dataDir, 'inpainted_data', 'true.csv')
    data_true = pd.read_csv(data_true_Dir)
    data_true_header_ls = data_true.columns.values
    # print(data_true)

    data_outdate_Dir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask_r{seed}', f'floor{floor}', f'month{old_month}',
                         f'swinT_mask{int(mask_ratio * 100)}', 'inpainted_data', 'true.csv')
    data_outdate = pd.read_csv(data_outdate_Dir)
    data_outdate_header_ls = data_outdate.columns.values

    # header_ls = extra_same_elem(data_update_header_ls, data_outdate_header_ls)

    # read mask
    mask = get_mask(seed)

    # local update
    start = time.time()
    update_local = sliding_window_renew(sampled_coord, data_um, data_update, data_outdate, mask, (2, 3), stride=1)
    end = time.time()
    print(f'use time (local): {end - start}s')

    # calculate error
    local_error_ls = calculate_error(update_local, data_true)
    aver_error = np.mean(local_error_ls)
    print("local update: ", aver_error)
    # print("local max error: ", max(error_ls))
    # print(len(error_ls))

    global_error_ls = calculate_error(data_update, data_true)
    aver_error = np.mean(global_error_ls)
    print("global update:", aver_error)


    # data_outdate_merge, data_true_merge = merge_outdateTrue_AP(data_outdate_header_ls, data_true_header_ls)
    # outdate_error_ls = calculate_error(data_outdate_merge, data_true_merge)
    # aver_error = np.mean(outdate_error_ls)
    # print("outdate update:", aver_error)

    # plot
    # plot_cdf(local_error_ls, global_error_ls)

    # plot error bar
    plot_err_bar(local_error_ls, global_error_ls, savePath="./")


if __name__ == '__main__':
    main()
