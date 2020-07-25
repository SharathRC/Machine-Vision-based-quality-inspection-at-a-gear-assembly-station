import cv2
import numpy as np
import pathlib
import sys

import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
import os
from os.path import isfile, join
# print(os.getcwd(), 'here')
# os.chdir(os.path.dirname(__file__))

AUGMENT_IMAGES = False
NUM_CLASSES = 2
CLASS_NAMES = ['complete', 'incomplete']

def fix_cad_img_size(cad_img, bkg):
    bkg_h, bkg_w, _ = bkg.shape
    cad_h, cad_w, _ = cad_img.shape

    diff_h = bkg_h - cad_h
    diff_w = bkg_w - cad_w
    
    if diff_h > 0:
        horizontal1 = np.zeros((int(diff_h/2), cad_img.shape[1],3), dtype=np.uint8)
        cad_img = np.append(horizontal1, cad_img, axis=0)
        if not diff_h%2==0:
            horizontal2 = np.zeros((int(diff_h/2)+1, cad_img.shape[1],3), dtype=np.uint8)
        else:
            horizontal2 = horizontal1
        cad_img = np.append(cad_img, horizontal2, axis=0)
    # if diff_h < 0:
    #     cad_img = image_resize(cad_img, height=bkg.shape[0])

    if diff_w > 0:
        vertical1 = np.zeros((cad_img.shape[0], int(diff_w/2), 3), dtype=np.uint8)
        cad_img = np.append(vertical1, cad_img, axis=1)
        if not diff_w % 2 ==0:
            vertical2 = np.zeros((cad_img.shape[0], int(diff_w/2)+1, 3), dtype=np.uint8)
        else:
            vertical2 = vertical1
        cad_img = np.append(cad_img, vertical2, axis=1)
    bb_box = (int(diff_w/2), int(diff_h/2), int(diff_w/2 + cad_w), int(diff_h/2 + cad_h))
    return cad_img, bb_box

def merge_bkg(cad_gray, cad_img, bkg):
    # mask = np.ones(cad_gray.shape, dtype=np.uint8)
    # mask[cad_gray==0] = 0
    # print(cad_gray.shape, cad_img.shape, bkg.shape)
    cad_img[cad_gray == 0] = bkg[cad_gray == 0]

    return cad_img

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def crop_cad_img(cad_img):
    _ ,cad_img = cv2.threshold(cad_img,40,255,cv2.THRESH_TOZERO)
    cad_gray = cv2.cvtColor(cad_img, cv2.COLOR_BGR2GRAY)
    _,cad_binary = cv2.threshold(cad_gray, 0, 255, cv2.THRESH_BINARY)
    r1, c1, r2, c2 = get_cad_dims(cad_binary)
    cad_img = cad_img[r1-5:r2+5, c1-5:c2+5]
    return cad_img

def get_cad_dims(cad_binary):
    r, c = np.where(cad_binary!=0)
    return min(r), min(c), max(r), max(c)

def show_img(img):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test', (1000,600))
    cv2.imshow('test', img)
    cv2.waitKey(0)

def start():
    cad_img = cv2.imread('../volumes/1.png')
    bkg = cv2.imread('../volumes/Backgrounds/bkg (1).jpeg')

    cad_img = crop_cad_img(cad_img)

    resize_height = int(0.5 * bkg.shape[0])
    cad_img = image_resize(cad_img, height = resize_height)

    cad_img, bkg = fix_cad_img_size(cad_img, bkg)
    cad_gray = cv2.cvtColor(cad_img, cv2.COLOR_BGR2GRAY)

    cad_img = merge_bkg(cad_gray, cad_img, bkg)

    # print(cad_img)

    cv2.imwrite('../volumes/test3.jpg', cad_img)

    show_img(cad_img)

def add_background(cad_img, bkg):
    cad_img, bb_box = fix_cad_img_size(cad_img, bkg)
    
    cad_gray = cv2.cvtColor(cad_img, cv2.COLOR_BGR2GRAY)

    cad_img = merge_bkg(cad_gray, cad_img, bkg)
    return cad_img, bb_box

def bb_box2list(bb_box, h, w):
    x1 = bb_box.x1
    y1 = bb_box.y1 
    x2 = bb_box.x2
    y2 = bb_box.y2
    
    return [x1, y1, x2, y2]

def correct_bb_box(bb_box, h, w):
    x1, y1, x2, y2 = bb_box[0], bb_box[1], bb_box[2], bb_box[3]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    if x1 > w:
        x1 = w
    if y1 > h:
        y1 = h
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h
    
    return [x1, y1, x2, y2]

def save_img(img, bb_box, filename, class_id):
    h, w, _ = img.shape
    x1, y1, x2, y2 = bb_box[0], bb_box[1], bb_box[2], bb_box[3]
    rel_bb_box_x_center = (x1 + x2)/(2*w)
    rel_bb_box_y_center = (y1 + y2)/(2*h)
    rel_bb_box_w = (x2 - x1)/w
    rel_bb_box_h = (y2 - y1)/h

    img_file = f'{filename}.jpg'
    text_file = f'{filename}.txt'
    with open(text_file, 'w') as f:
        f.write(f'{class_id} {rel_bb_box_x_center} {rel_bb_box_y_center} {rel_bb_box_w} {rel_bb_box_h}')
        f.close()
    cv2.imwrite(img_file, img)

def start_folder(bkg_mode=None):
    seq = iaa.Sequential([iaa.Affine(rotate=(-25, 25)),\
                            iaa.AdditiveGaussianNoise(scale=(5, 60)),\
                            iaa.Crop(percent=(0, 0.2))], random_order=True)

    if AUGMENT_IMAGES:
        save_directory = '../../volumes/augmented'
    else:
        save_directory = '../../volumes/added_background'
    
    for class_name in CLASS_NAMES:
        if not os.path.exists(f'{save_directory}/{class_name}'):
            os.makedirs(f'{save_directory}/{class_name}')

    bkg_folder = f'../../volumes/{bkg_mode}'
    
    bkg_type = f'_{bkg_mode}'
    
    for class_id in range(2):
        for perspective in range(1, 5):
            print(class_id, perspective)
            cad_img_name = f'c{class_id}_p{perspective}'
            try:
                cad_img_org = cv2.imread(f'../../volumes/cad_models/{cad_img_name}.png')
                cad_img_org = crop_cad_img(cad_img_org)
            except Exception as e:
                print(e)
                continue
            
            count = 0
            for path in pathlib.Path(f'{save_directory}/{CLASS_NAMES[class_id]}').iterdir():
                filename = str(path).split("\\")[-1]
                if path.is_file() and filename.endswith('jpg') and filename.startswith(f'{class_id}'):
                    count += 1

            bkg_images_list = []
            for path in pathlib.Path(bkg_folder).iterdir():
                filename = str(path).split("\\")[-1]
                bkg_images_list.append(f'{bkg_folder}/{filename}')

            for img_path in bkg_images_list:
                bkg = cv2.imread(img_path)

                resize_height = int(0.6 * bkg.shape[0])
                cad_img = image_resize(cad_img_org, height = resize_height)
                if cad_img.shape[1] > bkg.shape[1]:
                    resize_width = int(0.6 * bkg.shape[1])
                    cad_img = image_resize(cad_img_org, width=resize_width)
                    
                out_img, bb_box = add_background(cad_img, bkg)

                if AUGMENT_IMAGES:
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                    bbs = [[ia.BoundingBox(x1=bb_box[0], y1=bb_box[1], x2=bb_box[2], y2=bb_box[3])]]

                    imgs_and_bbs_set = [seq(image=out_img, bounding_boxes=bbs) for _ in range(8)]
                    for (img, bb_box) in imgs_and_bbs_set:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        h, w, _ = img.shape
                        
                        bb_box = bb_box[0][0]
                        bb_box = bb_box2list(bb_box)
                        bb_box = correct_bb_box(bb_box, h, w)

                        filename = f'{save_directory}/{CLASS_NAMES[class_id]}/{class_id}{bkg_type}_{count}'
                        save_img(img=img, bb_box=bb_box, filename=filename, class_id=class_id)
                        count+=1
                else:
                    h, w, _ = out_img.shape
                    bb_box = correct_bb_box(bb_box, h, w)
                    filename = f'{save_directory}/{CLASS_NAMES[class_id]}/{class_id}{bkg_type}_{count}'
                    save_img(img=out_img, bb_box=bb_box, filename=filename, class_id=class_id)
                    count+=1

def gen_images():
    bkg_modes = ['factory', 'concrete', 'metal', 'wood' ]
    # bkg_mode = ['concrete', 'metal', 'wood' ]
    for bkg in bkg_modes:
        start_folder(bkg)


# start_folder('wood')
gen_images()
print('done!')