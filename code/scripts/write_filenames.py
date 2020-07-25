from os import listdir
import os
from os.path import isfile, join
import pathlib
from random import shuffle
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = os.getcwd()
print(path)

folder_path = '../../volumes/added_background'
num_classes = 2

all_classes_imgs = []
for _ in range(num_classes):
    all_classes_imgs.append([])

data_dir = pathlib.Path(folder_path)
# img_list = list(data_dir.glob('*/*.jpg'))
img_list = list(data_dir.glob('*.jpg'))
image_count = len(img_list)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)


for path in img_list:
    img_name = str(path).split("/")[-1]
    img_name = str(path).split("\\")[-1]
    if img_name.endswith('jpg'):
        class_id = int(img_name.split('_')[0])
        all_classes_imgs[class_id].append(img_name)

trainfiles = []
valfiles = []
testfiles = []

for i in range(len(all_classes_imgs)):
    shuffle(all_classes_imgs[i])
    num_files = len(all_classes_imgs[i])
    trainfiles += all_classes_imgs[i][0:int(0.7*num_files)]
    valfiles += all_classes_imgs[i][int(0.7*num_files):]

shuffle(trainfiles)
shuffle(valfiles)

# add_path = '/darknet/custom/images'
# add_path = '/workspace/training_custom/images'
add_path = '/content/drive/My Drive/images'
# add_path = '../../volumes/images'
write_path = '../../volumes'
# write_path = 'finetune_alexnet_with_tensorflow/gear_objects'
# write_path = '../darknet/custom'

with open(f'{write_path}/train.txt', 'w') as f:
    for filename in trainfiles:
        # class_id = int(filename.split('_')[0])
        # f.write(f'{add_path}/{CLASS_NAMES[class_id]}/{filename}\n')
        f.write(f'{add_path}/{filename}\n')

with open(f'{write_path}/val.txt', 'w') as f:
    for filename in valfiles:
        # class_id = int(filename.split('_')[0])
        # f.write(f'{add_path}/{CLASS_NAMES[class_id]}/{filename}\n')
        f.write(f'{add_path}/{filename}\n')

# with open(f'{write_path}/test.txt', 'w') as f:
#     for filename in onlyfiles:
#         # class_id = int(filename.split('_')[0])
#         # f.write(f'{add_path}/{CLASS_NAMES[class_id]}/{filename}\n')
#         f.write(f'{add_path}/{filename}\n')