import cv2
import numpy as np
import os


def crop(img1, img2, img3, img4, img5, label, m, img_save_dir, label_save_dir):
    
    h, w = label.shape
    _, thresh = cv2.threshold(label, 20, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('binary', 0)
    # cv2.imshow('binary', thresh)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel_erode, iterations=1)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=4)  # 膨胀
    # cv2.namedWindow('b', 0)
    # cv2.imshow('b', thresh)
    # remove holes within cells, and make sure cropped images have unbroken features
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    new_label = np.zeros((h, w, 1), np.uint8)
    new_label.fill(255)
    j = 0
    cv2.waitKey(1)
    print('len(contours)', len(contours))
    for i in range(len(contours)):  # contours counts
        # print(hierarchy[0,i,3])
        area = cv2.contourArea(contours[i])
        if area > 2000:
            print('area=', area)
        if area > 1000 and hierarchy[0, i, 3] == -1:
            rect = cv2.minAreaRect(contours[i])
            cv2.drawContours(new_label, contours[i], -1, (0, 0, 0), 1)
            # cv2.namedWindow('nl',0)
            cv2.imshow('nl', new_label)
            cv2.waitKeyEx(1)
            box = np.int0(cv2.boxPoints(rect))  # return 4 position
            # draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 2)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            if x1 - 10 > 0 and y1 - 10 > 0 and x2 + 10 < h and y2 + 10 < w:
                crop_img1 = crop_rect(img1, rect)
                crop_img2 = crop_rect(img2, rect)
                crop_img3 = crop_rect(img3, rect)
                crop_img4 = crop_rect(img4, rect)
                crop_img5 = crop_rect(img5, rect)
                crop_label = crop_rect(label, rect)
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                if not os.path.exists(label_save_dir):
                    os.makedirs(label_save_dir)
                cv2.imwrite(os.path.join(img_save_dir, str(m) + '_' + str(j) + '.tif'), crop_img1)
                cv2.imwrite(os.path.join(img_save_dir, str(m + 1) + '_' + str(j) + '.tif'), crop_img2)
                cv2.imwrite(os.path.join(img_save_dir, str(m + 2) + '_' + str(j) + '.tif'), crop_img3)
                cv2.imwrite(os.path.join(img_save_dir, str(m + 3) + '_' + str(j) + '.tif'), crop_img4)
                cv2.imwrite(os.path.join(img_save_dir, str(m + 4) + '_' + str(j) + '.tif'), crop_img5)
                cv2.imwrite(os.path.join(label_save_dir, str(m) + '_' + str(j) + '.tif'), crop_label)
                j = j + 1


def crop_rect(img, rect):
    # get target parameter of  small rectangle
    print("rect!", rect)
    point = img[1, 1]
    try:
        if len(point) == 3:
            bb = int(point[0])
            gg = int(point[1])
            rr = int(point[2])
            borderValue = (bb, gg, rr)
    except:
        borderValue = None
    
    center, sizes, angle = rect[0], rect[1], rect[2]
    factor = 1.2
    size = tuple([factor * i for i in list(sizes)])
    if (angle > -45):
        center, size = tuple(map(int, center)), tuple(map(int, size))
        dnaight, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # size = tuple([int(rect[1][1]), int(rect[1][0])])
        img_rot = cv2.warpAffine(img, M, (width, dnaight), borderValue=borderValue)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        angle -= 270
        dnaight, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, dnaight), borderValue=borderValue)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop

def main(root_dir=None,
         material_dir=None,
         original_data_dir=None,
         flu_class=None):
    if len(flu_class) == 3:
        bright_or_dir = os.path.join(original_data_dir, 'bright')  # whole RGB images 明场路径 _jundnan
        label_dir = os.path.join(original_data_dir, 'actin_max')  # label for whole images
        actin_or_dir = os.path.join(original_data_dir, 'actin')  # label for 4 kinds of lesions
        dna_or_dir = os.path.join(original_data_dir, 'dna')
        or_label_dir = (bright_or_dir, actin_or_dir, dna_or_dir)
        
        label_crop_dir = os.path.join(material_dir, 'actin_max')
        bright_crop_dir = os.path.join(material_dir, 'bright')  # cropped label for 4 kinds of lesions
        actin_crop_dir = os.path.join(material_dir, 'actin')
        dna_crop_dir = os.path.join(material_dir, 'dna')
        crop_img_dir = (bright_crop_dir, actin_crop_dir, dna_crop_dir)  # 切割明场路径
    
    label_list = os.listdir(label_dir)
    for label_name in label_list:
        label = cv2.imread(os.path.join(label_dir, label_name), 0)
        lab_nm, extension = os.path.splitext(label_name)
        for i in range(len(flu_class)):
            print('i=', i)
            m = int(lab_nm)
            print(or_label_dir[i])
            s = os.path.join(or_label_dir[i], str(m) + '.tif')
            print(s)
            img1 = cv2.imread(os.path.join(or_label_dir[i], str(m) + '.tif'))
            img2 = cv2.imread(os.path.join(or_label_dir[i], str(m + 1) + '.tif'))
            img3 = cv2.imread(os.path.join(or_label_dir[i], str(m + 2) + '.tif'))
            img4 = cv2.imread(os.path.join(or_label_dir[i], str(m + 3) + '.tif'))
            img5 = cv2.imread(os.path.join(or_label_dir[i], str(m + 4) + '.tif'))
            crop(img1, img2, img3, img4, img5, label, m, crop_img_dir[i], label_crop_dir)
            


def creat_mask(workdir):
    indir = os.path.join(workdir, "actin")
    floder_list = os.listdir(indir)
    floder_list.sort(key=lambda x: int(x[:-4]))
    save_path = os.path.join(workdir, "actin_max")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for m in range(0, len(floder_list), 5):
        output_imgs = []
        for n in range(5):
            imgPath = indir + '/' + floder_list[m + n]  # '{}.tif'.format(k)
            image_3 = cv2.imread(imgPath)
            image = image_3[:, :, 2]
            H, W = image.shape
            output_imgs.append(image)
        img_zero = np.max(output_imgs, axis=0)
        img_save_name = save_path + '/' + floder_list[m]
        cv2.imwrite(img_save_name, img_zero)


def run_main(user_name=''):
    root_dir = os.path.join(r'./datasets', user_name)
    material_dir = os.path.join(root_dir, 'pbda', 'crop_materials')  # croped materials
    creat_mask(root_dir)
    main(root_dir=root_dir,
         material_dir=material_dir,
         original_data_dir=root_dir,
         flu_class=['dna', 'actin', 'bright'])


if __name__ == '__main__':
    run_main()
