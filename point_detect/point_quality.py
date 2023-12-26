from PIL import Image
from utils import *
from scipy import ndimage
import pandas as pd

np.set_printoptions(threshold=np.inf)

mask_ratio = 0.9
floor = 3
month = 1
badp_threshold = -90
goodp_threshold = -70

test_dir = oj('/nfs/UJI_LIB/data/updateDataset/', f'floor_{floor}', 'predict', f'month_{month}')
test_set = ls(test_dir)




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
    bad_points_num = np.sum(data <= threshold)
    points_num = data.shape[0] * data.shape[1]
    return bad_points_num / points_num


def get_C(data):
    pearson_matrix = np.corrcoef(data)
    pearson_ls = pearson_matrix[tuple(np.triu_indices(8, k=1))]
    C = (np.mean(pearson_ls) + 1) / 2
    return C


def get_rss(dataDir):
    data = pd.read_csv(dataDir)
    rss = np.array(data.iloc[:, 3:])
    return rss


def get_error(dataDir):
    with open(dataDir, 'r') as f:
        err = json.load(f)
    return err['mean_err']


def get_info(dataDir):

    # get mask
    img0 = Image.open(oj(test_dir, test_set[0]))
    img0 = transforms.ToTensor()(img0)
    img0 = torch.unsqueeze(img0, dim=0)

    mask = gen_input_mask_random(
            shape=(1, 1, img0.shape[2], img0.shape[3]),
            mask_ratio=mask_ratio
        )

    mask = np.array(mask, dtype=np.int32)

    mask[mask==1] = 2
    mask[mask==0] = 1
    mask[mask==2] = 0

    rssFileDir = oj(dataDir, 'mask_data', 'unmaskRss.csv')
    rss = get_rss(rssFileDir)

    D = get_D(mask)
    # get sampling RP fingerprint
    S = get_S(rss, badp_threshold)
    C = get_C(rss)
    errorFileDir = oj(dataDir, 'errData', 'err_result.json')
    error = get_error(errorFileDir)

    score = D * 0.5 * (S + C)  # 0.45
    return D, S, C, score, error


def main():
    dataDir = oj('/nfs/UJI_LIB/data/mask_test/', f'update_mask', f'floor{floor}', f'month{month}',
                 f'swinT_mask{int(mask_ratio * 100)}')
    D, Q, C, score, error = get_info(dataDir)

if __name__ == '__main__':
    main()
