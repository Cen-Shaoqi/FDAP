from PIL import Image
from utils import *
from scipy import ndimage
import pandas as pd

np.set_printoptions(threshold=np.inf)
# [7, 53, 81, 84, 102, 113, 162, 170, 178, 183, 218, 228, 245, 266, 307]
# [81, 84, ]
# seed = 2
# seed = 36
mask_ratio = 0.9
floor = 3
month = 1
badp_threshold = -90
goodp_threshold = -70
# setup_seed(seed)

test_dir = oj('/nfs/UJI_LIB/data/updateDataset/', f'floor_{floor}', 'predict', f'month_{month}')
test_set = ls(test_dir)
# dataDir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask_r{seed}', f'floor{floor}', f'month{month}', f'swinT_mask{int(mask_ratio*100)}')


# def gen_bad_mask_random(
#         shape, mask_ratio=0.5, mask_area=None):
#     random.seed(42)  # 保证随机结果可复现
#     mask = torch.zeros(shape)
#     bsize, _, mask_h, mask_w = mask.shape
#     token_count = mask_h * mask_w
#     sample_count = int(np.ceil(token_count * (1 - mask_ratio))) - 1
#     for i in range(bsize):
#         if mask_area is not None:
#             # print("mask_area is not None.")
#             mask[i, :] = mask[i, :] + mask_area
#         else:
#             # mask_idx = np.random.permutation(token_count)[:mask_count]
#             # single_mask = np.zeros(token_count)
#             front_mask_idx = np.random.permutation(int(token_count/4))[:sample_count]
#             front_mask = np.ones(int(token_count/4))
#             front_mask[front_mask_idx] = 0
#             back_mask = np.ones(token_count - int(token_count/4))
#             single_mask = np.concatenate((front_mask, back_mask))
#             single_mask = single_mask.reshape((mask_h, mask_w))
#
#             mask[i, :] = mask[i, :] + single_mask
#
#     return mask



def get_distance_aver(matrix, x, y):
    x_idx, y_idx = np.where(matrix == 1)
    points = np.transpose([x_idx, y_idx])
    distances = np.sqrt(np.sum(np.asarray(points - [x, y])**2, axis=1))
    return np.mean(distances)


def get_D(mask):
    # get center of mass
    _, _, x_center, y_center = ndimage.center_of_mass(mask)
    # print(x_center, y_center)

    # get aver of distance
    distance_aver = get_distance_aver(mask[0][0], x_center, y_center)

    # distance_aver / centerCount_point_distance
    # distance_cpoint = np.sqrt(np.sum((np.array([int(mask.shape[2]/2), int(mask.shape[3]/2)]) - np.array([x_center, y_center])) ** 2))
    distance_cpoint = np.sqrt(np.sum((np.array([int(mask.shape[2]/2), int(mask.shape[3]/2)]) - np.array([0.0, 0.0])) ** 2))
    # print(mask.shape)
    # print(int(mask.shape[2]/2), int(mask.shape[3]/2))
    d = distance_aver/distance_cpoint
    # d = 1 - d
    # if d > 1:
    #     d = 1 + np.finfo(float).eps
    return d


def cal_precent(fingerprint, threshold):
    bad_points_num = np.sum(fingerprint <= threshold)
    return bad_points_num / len(fingerprint)


def get_S(data, threshold):
    # 遍历每行，求占比
    precent_ls = []
    bad_points_num = np.sum(data <= threshold)
    # print(bad_points_num)
    points_num = data.shape[0] * data.shape[1]
    # for fingerprint in data:
    #     # print(fingerprint)
    #     precent_ls.append(cal_precent(fingerprint, threshold))
    # 求平均
    # Q = np.mean(precent_ls)
    return bad_points_num / points_num


def get_S2(data, threshold):
    # mu = np.mean(data, axis=0)
    # sigma = np.std(data, axis=0)
    # S = (data - mu) / sigma
    # data = data[data > threshold]
    good_points_num = np.sum(data > threshold)
    points_num = data.shape[0] * data.shape[1]
    good_count = good_points_num / points_num

    bad_points_num = np.sum(data <= badp_threshold)
    bad_count = bad_points_num / points_num
    print(bad_count)
    # print(good_count)
    # print(bad_count - good_count)



    # print(np.min(data))
    # print(np.max(data))
    # print(S)
    # return S
    # print(np.var(data))

def get_C(data):
    pearson_matrix = np.corrcoef(data)
    pearson_ls = pearson_matrix[tuple(np.triu_indices(8, k=1))]

    # f, ax = plt.subplots(figsize=(14, 10))
    # sns.set(font_scale=1.5)
    # sns.heatmap(pearson_matrix, linewidths=0.05, ax=ax, annot=True, cmap='Blues')
    # ax.set_title('Correlation between fingerprint')
    # plt.rc('font', family='Times New Roman', size=12)
    # plt.show()
    # plt.close()
    # 归一化 [-1, 1] -> [0, 1]
    C = (np.mean(pearson_ls) + 1) / 2
    return C


def get_rss(dataDir):
    data = pd.read_csv(dataDir)
    rss = np.array(data.iloc[:, 3:])
    return rss


def get_error(dataDir):
    with open(dataDir, 'r') as f:
        err = json.load(f)
        # print(err['mean_err'])
    return err['mean_err']


def get_info(dataDir, seed):
    setup_seed(seed)
    dataDir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask_r{seed}', f'floor{floor}', f'month{month}',
                 f'swinT_mask{int(mask_ratio * 100)}')

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

    rssFileDir = oj(dataDir, 'mask_data', 'unmaskRss.csv')
    rss = get_rss(rssFileDir)

    # >1 好
    D = get_D(mask)
    # D： 越高越好
    # D = 1 / (D + np.finfo(float).eps)
    # get sampling RP fingerprint
    # Q： 越低越好
    S = get_S(rss, badp_threshold)
    # get_S2(rss, goodp_threshold)
    # C: 越低越好 < 0.8 ？
    C = get_C(rss)
    errorFileDir = oj(dataDir, 'errData', 'err_result.json')
    error = get_error(errorFileDir)

    # score = D * Q * C
    # score = (D + Q + C) / 3.0  # 0.767
    score = D * 0.5 * (S + C)  # 0.45
    # print("D: ", D)
    # print("Q: ", Q)
    # print("C: ", C)
    # print("score: ", score)
    # print("error: ", error)
    return D, S, C, score, error


def main():
    # data = pd.DataFrame(columns=['D', 'Q', 'C', 'score', 'error', 'label'])
    data = []
    seed_ls1 = list(range(280))
    seed_ls2 = list(range(1000, 1263))
    seed_ls = seed_ls1 + seed_ls2
    # seed_ls = [0, 162]
    # seed_ls = list(range(604))
    for i, seed in enumerate(seed_ls):
        label = "good"
        dataDir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask_r{seed}', f'floor{floor}', f'month{month}',
                     f'swinT_mask{int(mask_ratio * 100)}')
        D, Q, C, score, error = get_info(dataDir, seed)
        if error >= 4.0:
            label = "bad"
        # data.append([D, Q, C, score, error, int(label)])
        data.append([D, Q, C, score, error, label])
        # print(result)
    data = pd.DataFrame(np.array(data))
    # data.to_csv('point_quality2.csv', header=['D', 'S', 'C', 'score', 'error', 'label'], index=None)
    # data.to_csv('point_quality3.csv', header=['D', 'Q', 'C', 'score', 'error', 'class'], index=None)
    # print("-------------------")
    # for i, seed in enumerate(seed_ls2):
    #     dataDir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask_r{seed}', f'floor{floor}', f'month{month}',
    #                  f'swinT_mask{int(mask_ratio * 100)}')
    #     D, Q, C, score, error = get_info(dataDir, seed)


if __name__ == '__main__':
    main()
