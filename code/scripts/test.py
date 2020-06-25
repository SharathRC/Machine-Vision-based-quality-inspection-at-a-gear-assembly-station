import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

import cv2

cv_img = cv2.imread('../assets/cad_models/c1_p1.png')
# cv_img = cv2.rectangle(cv_img, (600,50), (1000, 500), color=(0,255,0), thickness=4)
# cv2.imshow('test', cv_img)
# cv2.waitKey(0)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
bbs = [
    [ia.BoundingBox(x1=600, y1=50, x2=1000, y2=500)]
]

# seq = iaa.Sequential([
#     iaa.AdditiveGaussianNoise(scale=0.05*255),
#     iaa.Affine(translate_px={"x": (1, 5)})
# ])

seq = iaa.Sequential([iaa.Affine(rotate=(-25, 25)),\
                            iaa.AdditiveGaussianNoise(scale=(5, 60)),\
                            iaa.Crop(percent=(0, 0.2))], random_order=True)

                            # [seq(image=out_img) for _ in range(8)]
cv_img_aug, bbs_aug = seq(image=cv_img, bounding_boxes=bbs)
print(bbs_aug)
bb_box = bbs_aug[0][0]
x1 = bb_box.x1
y1 = bb_box.y1
x2 = bb_box.x2
y2 = bb_box.y2

if x1 < 0:
    x1 = 0
if y1 < 0:
    y1 = 0
if x2 < 0:
    x2 = 0
if y2 < 0:
    y2 = 0

cv2.rectangle(cv_img_aug, (x1, y1), (x2, y2), color=(0,255,0), thickness=4)
cv2.imshow('test', cv_img_aug)
cv2.waitKey(0)