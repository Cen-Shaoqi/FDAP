
def read_data(path, month):
    pass

def load_rssi(month, isUpdate=False):
    if isUpdate == False:
        # rssi = read_data(opt.dataPath, month)
        # return rssi
        return
    else:
        # rssi = read_data(updatePath, month)
        return
    pass

def kNN_ips(rssi):
    pass

# def update_data(isUpdate=True):
#     if isUpdate == False:
#         return
#     pass

def plot_err(err_ls):
    pass


if __name__ == '__main__':
    err_ls = []
    for i in range(1, 16):
        rssi_Mon1 = load_rssi(month=i, isUpdate=False)
        averIpsError_Mon1 = kNN_ips(rssi_Mon1)
        err_ls.append(averIpsError_Mon1)
        # update_data(isUpdate=False)
    plot_err(err_ls)

