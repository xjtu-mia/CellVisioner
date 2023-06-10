import cv2
import os

import numpy as np
import cv2
from PIL import Image


def crop(inpath, outpath):
    img = cv2.imread(inpath)
    x, y = 200, 400
    new_img = img[y: y + 512, x:x + 512]
    cv2.imwrite(outpath, new_img)


def cvt_rgb(indir, outdir, flu):
    """
    imgPath: 存放要加和的图片的文件夹的路径
    saveFolderPath: 加和后的图片保存的路径
    """
    # os.makedirs(saveFolderPath, exist_ok=True)
    if not os.path.exists(os.path.join(outdir, 'pred')):
        os.makedirs(os.path.join(outdir, 'pred'))
    if not os.path.exists(os.path.join(outdir, 'rgb')):
        os.makedirs(os.path.join(outdir, 'rgb'))
    
    imgList = os.listdir(indir)
    for img_name in imgList:
        imgPath1 = indir + '/' + img_name
        print(imgPath1)
        
        try:
            image1 = cv2.imread(imgPath1, 0)
            cv2.imwrite(os.path.join(outdir, 'pred', img_name), image1)
            out = np.zeros((image1.shape[0], image1.shape[1], 3), image1.dtype)
            if flu == 'actin':
                out[:, :, 2] = image1
            elif flu == 'dapi':
                out[:, :, 0] = image1
            else:
                print('error')
            
            save_name = os.path.join(outdir, 'rgb', img_name)
            cv2.imwrite(save_name, out)
        except:
            pass


if __name__ == '__main__':
    # 第一个参数改成存放要加和的图像的路径
    # 第二个参数改成保存加和后的图像的路径
    
    workdir = r'/data/zhanfengxiao/FRM_floder/result/231_20X/few_images/addimg'
    img_num = ['1', '2', '4', '8', '16', '32', '64', '128', '216']  # , '64', '128', '216'
    flu = 'dapi'
    data_aug = 'add_0.5_no_aug'
    for num in img_num:
        indir = os.path.join(workdir, num, flu, data_aug, 'result2', )
        outdir = os.path.join(workdir, 'rgb', num, f'{flu}_{data_aug}')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        cvt_rgb(indir, outdir, flu)
        crop(inpath=os.path.join(outdir, 'rgb', '311.tif'), outpath=os.path.join(outdir, 'crop_311.tif'))



