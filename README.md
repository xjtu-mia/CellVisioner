# CellVisioner
## A Generalizable Cell Virtual Staining Toolbox based on Few-Shot Transfer Learning for Mechanobiological Analysis ##
## 1. Environment
- Please prepare an environment with python>=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare cell datasets 
- You can download the datasets from the [Baidu Netdisk](https://pan.baidu.com/s/1NTQI-In5o9epd9itBq_gqA) (access code: kdgy)
* The data folder is named ./datasets/username/image_type, which comprises training images of brightfield, Actin, and DNA, as well as test images of brightfield. Each field of view consists of 5 layers of images collected at different depths. The details are as follows:
```bash
  .datasets
  ├── username
  |     ├── bright
  |     |     └── *.png
  |     └── actin
  |     |     └── *.png
  |     └── dna
  |     |     └── *.png
  |     └── test_bright
  |           └── *.png
          
```
## 3. Train and test the CellVisioner on your datasets
- Please modify the type of fluorescence images in main.py by selecting one or more options from "actin" and "dna" depending on your dataset. The functions ``` build_material_library.run_main(user_name) ``` and ``` possion_blend.run_main(user_name) ``` are only operable when the dataset contains both "actin" and "dna" image types. Otherwise, please comment them out.
-  ```options.py ``` contains various parameter settings that can be configured therein.
```bash
python main.py 
```

