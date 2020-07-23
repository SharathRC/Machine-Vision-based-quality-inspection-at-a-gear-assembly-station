import os
os.chdir(os.path.dirname(__file__))
print(os.getcwd())

with open('../../volumes/train.txt', 'r') as f:
    img_paths = []
    labels = []
    for path in f.readlines():
        img_paths.append(path[:-1])
        label = int(path.split('/')[-1].split('_')[0])
        labels.append(label)
    f.close()

print(img_paths)