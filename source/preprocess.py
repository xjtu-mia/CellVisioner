
import os

import cv2
from PIL import Image
import numpy as np
import time


def img_add_max(user_name, fluorescence):
    workdir = os.path.join('./datasets', user_name, 'pbda')
    for flu in fluorescence:
        input_path =os.path.join(workdir,f'{flu}')
        save_path = os.path.join(workdir,f'{flu}_add')
        input_list = os.listdir(input_path)
        input_list.sort(key=lambda x: int(x[:-4]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for m in range(0, len(input_list), 5):
            output_imgs = []
            for n in range(5):
                imgPath = input_path + '/' + input_list[m + n]
                image = cv2.imread(imgPath, 0)
                H, W = image.shape
                output_imgs.append(image)
            img_zero=np.max(output_imgs,axis=0)
            out = np.zeros((img_zero.shape[0],img_zero.shape[1]), img_zero.dtype)
            cv2.normalize(img_zero, out, 0, 255, cv2.NORM_MINMAX)
            img_save_name = save_path + '/' + input_list[m]
            cv2.imwrite(img_save_name, out)
  




def crop_and_save(input_dir, save_path, input_image, img_size_i):

    img=cv2.imread(os.path.join(input_dir , input_image))
    im_width, im_height = img.shape[0],img.shape[1]
    if im_width<img_size_i or im_height<img_size_i:
        min_size=min( im_width,im_height)
        img = cv2.resize(img, dsize=None, fx=np.cell(img_size_i/min_size),
                            fy=np.cell(img_size_i/min_size), interpolation=cv2.INTER_LINEAR)
        im_width, im_height = img.shape[0], img.shape[1]
        
    M = img_size_i
    N = img_size_i
    overlap_w = int((np.ceil(im_width / M) * M - im_width) / (np.ceil(im_width / M) - 1+1e-6)) + 1
    overlap_h = int((np.ceil(im_height / N) * N - im_height) / (np.ceil(im_height / N) - 1+1e-6)) + 1
    i=0
    ext = os.path.splitext(input_image)
    for w, x in enumerate(range(0, im_width, M)):
        for h, y in enumerate(range(0, im_height, N)):
            i += 1
            left = x if x == 0 else x - overlap_w * w
            up = y if y == 0 else y - overlap_h * h
            tiles = img[left: left + M, up:up + N]
            save_name=os.path.join(save_path,f'{int(ext[0])+i*10000}{ext[1]}')
            cv2.imwrite(save_name,tiles)

def crop_img(working_dir,img_size_i,fluorescence,train_or_val):

    input_dir=[os.path.join(working_dir,'bright')]
    for flu in fluorescence:
        input_dir.append(os.path.join(working_dir,f'{flu}_add'))

    outdir = [os.path.join(working_dir,f'{train_or_val}', 'bright')]
    for flu in fluorescence:
        outdir.append(os.path.join(working_dir,f'{train_or_val}',f'{flu}_add'))
    for dirs in outdir:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    print('-------------------------------------------')
    img_list = [sorted(os.listdir(dirs)) for dirs in input_dir]
    for i in range(len(img_list[0])):
        crop_and_save(input_dir[0], outdir[0], img_list[0][i],img_size_i)

    for cnt,flu in enumerate(fluorescence):
        for i in range(len(img_list[0])//5):
            crop_and_save(input_dir[cnt+1], outdir[cnt+1], img_list[cnt+1][i], img_size_i)



    return

def seg_main(user_name,fluorescence):
    model_size=512
    workdir_dir=os.path.join('./datasets',user_name,'pbda')
    crop_img(workdir_dir, model_size,fluorescence, train_or_val='train')
        
if __name__=='__main__':
    dirs=r''

    cell=[4,8,16,32,64,128,216]
    for name in cell:
        workdir = os.path.join(dirs,str(name),'pbda')
        print(workdir)
        # main(workdir)