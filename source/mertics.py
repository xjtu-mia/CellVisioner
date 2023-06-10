import os
import pandas as pd
from skimage.metrics import structural_similarity as  compare_ssim
import numpy as np
import cv2
from PIL import Image


def SSIM(img1, img2):
    # 计算两个灰度图像之间的结构相似度
    (score, diff) = compare_ssim(img1, img2, gaussian_weights=True, data_range=255,
                                 full=True)  # 滑动窗口大小101 , gaussian_weights=True , win_size=11  data_range=255,
    # (score, diff) = compare_ssim(img1, img2, multichannel=True, gaussian_weights=True, full=True) # 采用高斯加权计算每一窗口的均值、方差以及协方差，sigma=1.5
    # data[i] = score
    return score


def pearson(image1, image2):
    '''皮尔逊相关系数'''
    X = np.vstack([image1, image2])
    return np.corrcoef(X)[0][1]
def compute_mertics(data_root, gt_dir, pred_dir, flu):
    fl = os.listdir(gt_dir)
    i = 0
    pcc_all = 0
    ssim_all = 0
    a = []
    b = []
    c = []
    data = {}
    for file in fl:
        # print(file)
        i = i + 1
        image1_path = gt_dir + '/' + file
        image2_path = pred_dir + '/' + file
        
        image1 = cv2.imread(image1_path, 0)
        image2 = cv2.imread(image2_path, 0)
        ssim = SSIM(image1, image2)
        image1 = Image.fromarray(np.uint8(image1))
        image2 = Image.fromarray(np.uint8(image2))
        image1 = image1.resize(image2.size)
        image1 = np.asarray(image1).flatten()
        image2 = np.asarray(image2).flatten()
        pcc = pearson(image1, image2)
        
        # jcd = jaccard(image1, image2)
        pcc_all = pcc_all + pcc
        ssim_all = ssim_all + ssim
        ssim = round(ssim, 4)
        pcc = round(pcc, 4)
        data['name'] = a.append(file)
        data['pcc'] = b.append(pcc)
        data['ssim'] = c.append(ssim)
        # print(file,f'pcc={pcc},ssim={ssim}')
        # print(pearson(image2, image3))
    pcc_avg = pcc_all / i
    ssim_avg = ssim_all / i
    pcc_std = np.std(b)
    ssim_std = np.std(c)
    pcc_avg = round(pcc_avg, 4)  # 保留小数点后四位
    pcc_std = round(pcc_std, 4)
    ssim_avg = round(ssim_avg, 4)
    ssim_std = round(ssim_std, 4)
    print(f'pcc_avg={pcc_avg},pcc_std={pcc_std}')
    print(f'ssim_avg={ssim_avg},ssim_std={ssim_std}')
    data['name'] = a.append('avg')
    data['pcc'] = b.append(pcc_avg)
    data['ssim'] = c.append(ssim_avg)
    data['name'] = a.append('std')
    data['pcc'] = b.append(pcc_std)
    data['ssim'] = c.append(ssim_std)
    
    df = pd.DataFrame({'name': a, 'pcc': b, 'ssim': c})
    save_path = data_root + '/' + f'{flu}_pcc_and_ssim.csv'  # _denoise
    df.to_csv(path_or_buf=save_path, sep=',', na_rep='NA',
              columns=['name', 'pcc', 'ssim'])  # headers = False(不保存列名) index = False(不保存索引)


if __name__=="__main__":
    save_dir = './result'
    pb_aug ='no_aug'
    model_name ='cgan'
    
    cell_and_magnification = '231_20X/'
    fluorescence = 'dapi'
    number = ['4', '8', '16', '32', '64', '128', '216']
    for num in number:
        print(os.path.join(save_dir, cell_and_magnification, f'few_images/{num}',
                                       fluorescence, model_name, 'pretrained'))
        for i in range(5):
            result_path = os.path.join(save_dir, cell_and_magnification, f'few_images/{num}',
                                       fluorescence, model_name, 'pretrained', f'result{i}')
            # if not os.path.exists(result_path):
            #     os.makedirs(result_path)
            

            compute_mertics(result_path, os.path.join(result_path,'test', 'gt'),
                            os.path.join(result_path,'test', 'pred'), 'd')
