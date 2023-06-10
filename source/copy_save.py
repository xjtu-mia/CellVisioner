import cv2
import os
def copy_save(inpath,outpath):
    inlist=os.listdir(inpath)
    print(inlist)
    inlist.sort()
    

    for i in range(1,len(inlist)+1):
        imgpath=os.path.join(inpath,inlist[i-1])
        # print(imgpath)
        # img = cv2.imread(imgpath)
        ext=os.path.splitext(imgpath)[-1]
        save_name=os.path.join(outpath,str(i)+ext)
        # save_name=os.path.join(outpath,inlist[i-1])  #不修改名字
        # cv2.imwrite(save_name,img)
        os.rename(imgpath,save_name)

def train_datasets(user_name,fluorescence):
    path = os.path.join('./datasets', user_name, "datasets")
    save_path=os.path.join('./datasets', user_name,'no_aug')
    name_cell=['bright']
    name_cell.extend(fluorescence)
    # for i in range(10):
    #   print(name_cell)
    for name in name_cell:
        inpath=os.path.join(path,name)
        outpath = os.path.join(save_path, name)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        copy_save(inpath,outpath)

def test_datasets(user_name,cell_type):
    path = os.path.join(r'./datasets', user_name)

    inpath=os.path.join(path,"datasets",'test_bright')

    outpath = os.path.join(path,'test')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    copy_save(inpath,outpath)
if __name__=="__main__":
    path=r'E:\FRM\data\me\231_20X2\datasets'
    train_datasets(path)
    test_datasets(path)

