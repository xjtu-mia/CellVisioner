from source import build_material_library,possion_blend
from  source.model_select import model_select_main
import os
from train_unet import unet_main
from train_cgan import cgan_main
from predict import predict_main
from source.preprocess import  seg_main,img_add_max



def train(user_name, fluorescence):
    print("*************************************")
    print('build material library')
    build_material_library.run_main(user_name)
    print("*************************************")
    print('possion blend')
    possion_blend.run_main(user_name)
    print("*************************************")
    img_add_max(user_name,fluorescence)
    print('seg train datasets')
    seg_main(user_name,fluorescence)   # Segment image to 512*512 pixel
    print('model selecting')
    weight_name=model_select_main(user_name)   # Select best similar model weight
    print(f'the best matching cell is {weight_name}')
    print('start unet training')
    for flu in fluorescence:
        unet_main(user_name,flu,weight_name)
    print("*************************************")
    print('start cGAN training')
    for flu in fluorescence:
       cgan_main(user_name,flu,weight_name)

        
def test(user_name,  fluorescence):
 
    print('start unet predict')
    for flu in fluorescence:
        predict_main(user_name,flu,'unet')
    print('start cGAN predict')
    for flu in fluorescence:
        predict_main(user_name, flu, 'cgan')
    print('results of visualization ')


if __name__ == '__main__':
    user_name='CellVisioner'
    fluorescence=['actin' ,'dna']
    train(user_name,fluorescence)
    test(user_name,fluorescence)
