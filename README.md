# Text4VPR

This is the official repository for Text4VPR.



## Introduction

We focus on the localization problem from pure text to images, specifically achieving accurate positioning through descriptions of the surrounding environment. Our text4VPR model addresses this issue for the first time by utilizing semantic information from multiple views of the same location. During the training phase, we employ contrastive learning with single image-text pairs, while in the inference phase, we match groups of descriptions and images from the same location to achieve precise localization. We are the first to tackle the localization problem from pure text descriptions to image groups and have introduced a dataset called Street360Loc. This dataset contains 7,000 locations, each with four images from different directions and corresponding rich textual descriptions. On Street360Loc, Text4VPR builds a robust baseline, achieving a top-1 accuracy of 51.2% and a top-10 accuracy of 90.5% within a 5-meter radius on the test set. This indicates that localization from textual descriptions to images is not only feasible but also holds significant potential for further advancement.

## Building Environment

Create a conda environment and install basic dependencies:

```
git clone https://github.com/nuozimiaowu/Text4VPR
cd Text4VPR

conda create -n text4vpr python=3.11.9
conda activate text4vpr

# Install the according versions of torch and torchvision
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies
conda install --file requirements.txt
```

## Dataset Construction

"Download `Street360Loc_images.rar` from https://drive.google.com/file/d/17QlkGvgAKIYlm6AHi6fjFTD8ev8eNVWB/view?usp=sharing. Extract it into the `dataset` folder. The structure of the `dataset` folder will be as follows:"

dataset/
├── test_description.xlsx
├── train_description.xlsx
├── val_description.xlsx
└── Street360Loc_images/
     ├── 000001_0.jpg
     ├── 000001_1.jpg
     ├── 000001_2.jpg
     ├── 000001_3.jpg
     ..............
     ..............
     ├── 007000_5.jpg


Dataset construction finished



## Training

Open `train/train.py` In the following code, replace the paths that need to be converted with the paths from your dataset.

```
excel_file_train = r"your path to /dataset/train_description.xlsx"
excel_file_val = r"your path to /dataset/val_description.xlsx"
image_root_dir = r"your path to /dataset/Street360Loc_images"
```

After running **`train.py`**, the model is trained, and the model is saved in the `train/checkpoints` folder

## Evaluation

Open `test/test.py`. In the following code, replace the paths that need to be converted with the paths from your dataset.

```
excel_file_test = r" your path to /dataset/test_description.xlsx"
image_root_dir = r"your path to /dataset/Street360Loc_images"
```

Then, replace the weight file path in test.py :

```
model_path = r'your weight file path under train/checkpoints/'  
```

After running **`test.py`**,the model is evaluated.

