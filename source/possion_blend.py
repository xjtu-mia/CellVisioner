import cv2
import numpy as np
import os
import random
import math
from math import fabs, sin, cos, radians
from scipy.stats import mode
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os.path
import re


def blend(cr_patch, background, mask, center,  blend_strategy='possion_blend', type=cv2.MIXED_CLONE):  #cv2.MIXED_CLONE  cv2.NORMAL_CLONE cv2.MONOCHROME_TRANSFER
    h, w = mask.shape
    if blend_strategy == 'possion_blend':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_d = cv2.dilate(mask, kernel, iterations=8)
    
        # dilate flu masks to make sure that possion blending can not influence interior of flus excessively
        mixed_img = cv2.seamlessClone(cr_patch, background, mask_d, center, type)

    return mixed_img  #, mixed_label


def aug_data(img, label,
             rescale=False,
             rot=False,
             fl=False,
             elastic_trans=False,
             rescale_rate=1,
             degree=0,
             flipCode=0,
             filled_color=-1):
    def zoom(img, rate):
        h, w = img.shape[1], img.shape[0]
        new_h = round(h * rate)
        new_w = round(w * rate)
        if len(img.shape) == 3:
            inter = cv2.INTER_LINEAR
        else:
            inter = cv2.INTER_NEAREST
        new_img = cv2.resize(img, (new_h, new_w), interpolation=inter)
        # print(new_img.shape,'n',img.shape)
        return new_img

    def rotation(img, degree, filled_color=filled_color):  # rotation
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if filled_color == -1:
            filled_color = mode([img[0, 0], img[0, -1],
                                 img[-1, 0], img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])
        dnaight, width = img.shape[:2]
        # 旋转后的尺寸
        dnaight_new = int(width * fabs(sin(radians(degree))) +
                         dnaight * fabs(cos(radians(degree))))
        width_new = int(dnaight * fabs(sin(radians(degree))) +
                        width * fabs(cos(radians(degree))))
        mat_rotation = cv2.getRotationMatrix2D((width / 2, dnaight / 2), degree, 1)
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (dnaight_new - dnaight) / 2
        # Pay attention to tdna type of elements of filler_color, which should be
        # tdna int in pure python, instead of those in numpy.
        img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, dnaight_new),
                                     borderValue=filled_color)
        # 填充四个角
        mask = np.zeros((dnaight_new + 2, width_new + 2), np.uint8)
        mask[:] = 0
        seed_points = [(0, 0), (0, dnaight_new - 1), (width_new - 1, 0),
                       (width_new - 1, dnaight_new - 1)]
        for i in seed_points:
            cv2.floodFill(img_rotated, mask, i, filled_color)
        if len(img_rotated.shape) == 2:
            _, img_rotated = cv2.threshold(img_rotated, 127, 255, cv2.THRESH_BINARY)
        return img_rotated

    def flip(img, flipCode=flipCode):  # flip
        flip_img = cv2.flip(img, flipCode)
        return flip_img

    def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
        # 其中center_square是图像的中心，square_size=512//3=170
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
        M = cv2.getAffineTransform(pts1, pts2)  # 获取变换矩阵
        # 默认使用 双线性插值，
        image[:, :, :] = cv2.warpAffine(image[:, :, :], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT)
        # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
      
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)  # 构造一个尺寸与dx相同的O矩阵
        # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 网格采样点函数
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return map_coordinates(image, indices, order=0, mode='nearest').reshape(shape)

    # for label
    if rescale:
        #img = zoom(img, rate=rescale_rate)
        label = zoom(label, rate=rescale_rate)
    if fl:
        #img = flip(img, flipCode=flipCode)
        label = flip(label, flipCode=flipCode)
    if rot:
        #img = rotation(img, degree, filled_color=filled_color)
        label = rotation(label, degree, filled_color=filled_color)
    #for img
    for i in range(15):
        image=img[i]
        if rescale:
            image = zoom(image, rate=rescale_rate)
            # label = zoom(label, rate=rescale_rate)
        if fl:
            image = flip(image, flipCode=flipCode)
            # label = flip(label, flipCode=flipCode)
        if rot:
            image = rotation(image, degree, filled_color=filled_color)
            # label = rotation(label, degree, filled_color=filled_color)
        img[i]=image

    if elastic_trans:

        if elastic_trans:
            im1,im2,im3,im4,im5,im6,im7,im8,im9,im10,im11,im12,im13,im14,im15=np.array_split(img,15, axis=0)

            im_merge = np.concatenate((im1,im2,im3,im4,im5,im6,im7,im8,im9,im10,im11,im12,im13,im14,im15),axis=3)
            im_merge=im_merge[0,:,:,:]

            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 0.5, im_merge.shape[1] * 0.08,  #*im_merge.shape[1] * 2
                                           im_merge.shape[1] * 0.08)
            # Split image and mask
            n=0
            for i in range(45,3):
                img[n,:,:]=im_merge_t[:,:,i:i+3]
                n=n+1
  
        else:
            im1 = img
            la1 = label
            im_merge = np.concatenate((im1[..., None],
                                       la1[..., None]), axis=2)
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                           im_merge.shape[1] * 0.08)
            # Split image and mask
            img = im_merge_t[..., 0]
            label = im_merge_t[..., 1]
    return img, label


# 计算class_weight
def class_count(path, class_num):
    fl = os.listdir(path)
    p = np.zeros(class_num)
    total = np.zeros(class_num)
    n = 0
    for fn in fl:
        img = cv2.imread(os.path.join(path, fn), 0)
        img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4), interpolation=0)  # 减小计算量
        h, w = img.shape
        img = img.reshape(h * w)
        for i in range(class_num):
            p[i] = sum(img == i)
        for m in range(len(p)):
            total[m] = total[m] + p[m]
        print(fn)
    most = max(total)
    percentage = total / h / w / len(fl)
    for m in range(len(total)):
        total[m] = most / total[m]
    print(class_num)
    print('class-weight:', total, '\nclass-percentage:', percentage)


def main(flu_class=None,
         dens_dict=None,
         random_dens=None,
         aug=False,
         aug_rate=1,
         type=cv2.NORMAL_CLONE,
         original_data_dir=None,
         save_dir=None,
         material_dir=None):
    # origin image path
    label_dir = os.path.join(original_data_dir, 'actin_max')  # binary actin as mask
    bright_or_dir = os.path.join(original_data_dir, 'bright')
    actin_or_dir = os.path.join(original_data_dir, 'actin')  # gray actin
    dna_or_dir = os.path.join(original_data_dir, 'dna')
    or_yuan_dir = (bright_or_dir, actin_or_dir, dna_or_dir)
    # crop patch path
    cr_label_dir = os.path.join(material_dir, 'actin_max')
    cr_bright_dir = os.path.join(material_dir, 'bright')
    cr_actin_dir = os.path.join(material_dir, 'actin')
    cr_dna_dir = os.path.join(material_dir, 'dna')
    cr_img_dir = (cr_bright_dir,  cr_actin_dir, cr_dna_dir)
    # pbda results path
    blended_label_dir = os.path.join(save_dir, 'actin_max')
    blended_bright_dir = os.path.join(save_dir, 'bright')
    blended_actin_dir = os.path.join(save_dir, 'actin')
    blended_dna_dir = os.path.join(save_dir, 'dna')
    blend_img_dir = (blended_bright_dir, blended_actin_dir, blended_dna_dir)



    if not os.path.exists(blended_bright_dir):
        os.makedirs(blended_bright_dir)
    if not os.path.exists(blended_label_dir):
        os.makedirs(blended_label_dir)
    if not os.path.exists(blended_actin_dir):
        os.makedirs(blended_actin_dir)
    if not os.path.exists(blended_dna_dir):
        os.makedirs(blended_dna_dir)


    or_label_list = os.listdir(label_dir)
    print(or_label_list)
    
    for r in range(0, aug_rate + 0, 1):
        for or_label_fullname in or_label_list:  #actin_max
            print(or_label_fullname)
            or_img_name, extension = os.path.splitext(or_label_fullname)
            label = cv2.imread(os.path.join(label_dir, or_label_fullname))
            label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            _, label_binary = cv2.threshold(label_gray, 10, 255, cv2.THRESH_BINARY)

            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            label_binary = cv2.erode(label_binary, kernel1, iterations=2)
            label_binary = cv2.dilate(label_binary, kernel2, iterations=2)
          
            or_yuan_img=[]
            label_num = int(or_img_name)
            for i in range(len(flu_class)):
                for j in range(5):
                    orimgPath = or_yuan_dir[i] + '/' + str(label_num+j)+ extension
                    image = cv2.imread(orimgPath)
                    or_yuan_img.append(image)
    

            H, W = label.shape[:2]
            or_img_name=int(or_img_name)
            blended_img_name = str(1000*r+or_img_name) + extension
            dens = random.randint(random_dens[0], random_dens[1])
            print('dens=: ', dens)
      
            cr_label_list = os.listdir(cr_label_dir)
            n = 0
            patch_num=0
            while n < dens:
                patch_num+=1
            
                if patch_num==int(len(cr_label_list)/2):
                    print(' no appropriate patch')
                    break
                r_name = random.randint(0, len(cr_label_list) - 1)
                cr_label_name, extensions = os.path.splitext(cr_label_list[r_name])
                cr_label_num, cr_label_crop_num = re.split('_', cr_label_name)
                cr_label = cv2.imread(os.path.join(cr_label_dir, cr_label_list[r_name]),0)
           
                cr_label_num=int(cr_label_num)
                print("new patch")

                cr_img=[]
                for i in range(len(flu_class)):  # class
                   
                    for j in range(5):
                        crimgPath = cr_img_dir[i] + '/' + str(cr_label_num + j)+'_'+cr_label_crop_num+ extension
                        image = cv2.imread(crimgPath)
                        cr_img.append(image)


                if aug:
                    trigger = random.randint(1, 1)
                    if trigger == 1:
                        hp,wp=cr_label.shape
                        print(hp*wp)
                        if hp*wp>50000: #3T3_40X =90000
                            rescale_rate = random.uniform(0.6, 0.7)
                        elif 50000>=hp*wp>10000:  #3T3_40X =90000-10000
                            rescale_rate = random.uniform(0.7, 0.9)
                        else:
                            rescale_rate = random.uniform(0.9, 1.2)
                        degree = random.randrange(0, 360, 90) # 角度
                        flipCode = random.randint(-1, 1)  # 在范围内随机生成随机一个整数，闭区间
                        cr_img, cr_label = aug_data(cr_img,
                                                    cr_label,
                                                    rescale=True,
                                                    rot=False,
                                                    fl=True,
                                                    elastic_trans=False,
                                                    rescale_rate=rescale_rate,
                                                    degree=degree,
                                                    flipCode=flipCode,
                                                    filled_color=-1)

                h, w = cr_label.shape
                print(h * w)
                for time in range(100):
                   
                    r_h = random.randint(10 + math.ceil(h / 2), H-10 - math.ceil(h / 2))  #ceil() 函数返回数字的上入整数。
                    r_w = random.randint(10 + math.ceil(w / 2), W-10 - math.ceil(w / 2))  #取中心位置，且避免边界
                    
                    center = (r_w, r_h)
                    if np.max(label_binary[r_h - math.ceil(h / 2):r_h + math.ceil(h / 2),  #背景处不为空 or 背景与标签处不全为0
                              r_w - math.ceil(w / 2):r_w + math.ceil(w / 2)]) != 0 or \
                            np.max(cv2.bitwise_and(label_binary[r_h - math.ceil(h / 2):r_h + int(h / 2),
                                r_w - math.ceil(w / 2):r_w + int(w / 2)], cr_label, mask=cr_label) != 0):
                        print('Location is not suitable')


                    else:
                        n = n + 1
                        print('compete!')
                        ret, mask = cv2.threshold(cr_label, 0, 255, cv2.THRESH_BINARY)
                        # binary label
                        cr_label = cv2.cvtColor(cr_label, cv2.COLOR_GRAY2BGR)
                        label_binary = cv2.cvtColor(label_binary, cv2.COLOR_GRAY2BGR)
                        blended_label = blend(cr_label, label_binary, mask, center, type=type)  #泊松融合的函数必须为rgb图像
                        cr_label = cv2.cvtColor(cr_label, cv2.COLOR_BGR2GRAY)

                        blended_label = cv2.cvtColor(blended_label, cv2.COLOR_BGR2GRAY)
                        _, blended_label = cv2.threshold(blended_label, 0, 255,cv2.THRESH_BINARY)
                        label_binary=blended_label
                        #img
                        blended_img=[]
                        for i in range(15):
                            blended_image= blend(cr_img[i], or_yuan_img[i], mask, center, type=type)
                            blended_img.append(blended_image)
                        or_yuan_img=blended_img
                        break


            blended_img_num=0
            for i in range(len(flu_class)):  # 类别
                for j in range(5):
                    blend_img_Path = blend_img_dir[i] + '/' + str(1000*(r+1)+label_num + j) + extension #+'_' + str(r)
                    cv2.imwrite(blend_img_Path,or_yuan_img[blended_img_num])
                    blended_img_num = blended_img_num+1


            cv2.imwrite(os.path.join(blended_label_dir, blended_img_name), label_binary)
 

def run_main(user_name=''):
    root_dir = os.path.join(r'./datasets', user_name)
    material_dir = os.path.join(root_dir, 'pbda', 'crop_materials')
    original_data_dir = os.path.join(root_dir)
    save_dir = os.path.join(root_dir, 'pbda')

    main(flu_class=['bright', 'actin', 'dapi'],
         random_dens=[10, 15],
         aug=True,
         aug_rate=5,
         type=cv2.NORMAL_CLONE,
         material_dir=material_dir,
         save_dir=save_dir,
         original_data_dir=original_data_dir)

if __name__ == '__main__':
    run_main()




