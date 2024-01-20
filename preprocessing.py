import cv2
import numpy as np
import os
import shutil
import math
from argparse import ArgumentParser
from utils import *


def crop_images(
                path_to_imgs: str, 
                path_to_save: str,
                path_to_anns = None,
                crop_size: int = 500,
                stride:bool = False):
    '''
    Images in original format are of shape (more or less) 21000x2000. The central 
    ares is the only one of interest, so we crop each image into vertical
    patches and save them in a new folder. Same for annotations. 
    
    :param path_to_imgs: path to folder with images.
    :param path_to_anns: path to folder with annotations.
    :param path_to_save: path to folder where to save cropped images and annotations.
    :stride: if true when cropping the images we use a stride of 250px.

    '''
    if path_to_anns is None:
        path_to_anns = path_to_imgs
    else:
        path_to_anns = path_to_anns + '/anns'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        os.makedirs(os.path.join(path_to_save, "img"))
        os.makedirs(os.path.join(path_to_save, "ann"))

    cur_list = []
    for img_name, ann_name in zip(sorted(os.listdir(path_to_imgs)), sorted(os.listdir(path_to_anns))):
        img_id = img_name.split('_')[0]
        ann_id = ann_name.split('_')[0]

        assert img_id == ann_id, "Image and annotation ids must be the same"

        img_path = os.path.join(path_to_imgs, img_name)
        ann_path = os.path.join(path_to_anns, ann_name)

        cur_list.append(    
            crop(img_path, img_id, os.path.join(path_to_save, "img"), crop_size,stride)
        )

        crop(ann_path, img_id, os.path.join(path_to_save, "ann"), crop_size,stride)


def crop(image_path, img_id, save_path, crop_size,stride):
   
    if isinstance(crop_size, tuple):
        sub_image_height, sub_image_width = crop_size
    else:
        sub_image_height = crop_size
        sub_image_width  = crop_size

    img = cv2.imread(image_path)

    x = int((img.shape[1]/2) - (sub_image_width//2)) # center the crop
    
    y = 0 # start y
    crop_number = 0
    
    curated_list = []
    while y + crop_size <= img.shape[0]:
        
        image_crop = img[y:y+sub_image_height, x:x+sub_image_width]
        

        crop_number += 1
        id_sub_image = img_id + '_' + str(crop_number).zfill(3)
        curated_list.append(id_sub_image)
        img_name = os.path.join(save_path,id_sub_image + '.png')
        cv2.imwrite(img_name, image_crop)

        if stride:
            y += sub_image_height // 2  # Apply stride of half the image height
        else:
            y += sub_image_height  # Move 500 pixels down

    return curated_list


def split_crops(
                path_to_cropped, 
                path_to_save, 
                train_size: float = 0.6,
                ):
    '''
    Reorganize images and annotations in the CAUSE dataset structure. Also the 
    curated file is created. 

    :param path_to_cropped: path to folder with cropped images and annotations.
    :param path_to_save   : path to folder where to save cropped images and annotations.
    :param train_size     : train size for the split of the dataset.
    '''

    imgs_path = os.path.join(path_to_cropped, "img")
    anns_path = os.path.join(path_to_cropped, "ann")

    curated_path = os.path.join(path_to_save, "curated/train")
    imgs_ids = [string.split('_')[0] for string in os.listdir(imgs_path)]
    
    unique_imgs_id = sorted(np.unique(imgs_ids))
    
    train_imgs_id = unique_imgs_id[:math.ceil(len(unique_imgs_id)*train_size)]
    val_imgs_id = unique_imgs_id[math.ceil(len(unique_imgs_id)*train_size):]

    cur_file = open(os.path.join(curated_path, "curated.txt"), 'w')

    for img_name, anns_name in zip(os.listdir(imgs_path), os.listdir(anns_path)):

        img_id = img_name.split('_')[0]

        if img_id in train_imgs_id:
            shutil.copy(os.path.join(imgs_path, img_name), os.path.join(path_to_save, "img/train"))
            shutil.copy(os.path.join(anns_path, anns_name), os.path.join(path_to_save, "label/train"))

            cur_file.write(img_name + '\n')

        elif img_id in val_imgs_id:
            shutil.copy(os.path.join(imgs_path, img_name), os.path.join(path_to_save, "img/val"))
            shutil.copy(os.path.join(anns_path, anns_name), os.path.join(path_to_save, "label/val"))

    cur_file.close()


def main(
        path_to_json: str,
        path_to_imgs: str,
        path_to_anns,
        dataset_name: str,
        crop_size: int = 500,
        diversify_string: bool = True,
        ref_imgs: bool = False,
        stride:bool =False,
        train_size: float = 0.6
        ):
    
    path_to_crops = os.path.join("crops") 

    # Reformat images names to the format id_nameimg.png
    if ref_imgs:
        reformat_imgs(path_to_imgs)

    if path_to_anns is None:
        ann_imgs_path = 'annotations' 
    else:
        ann_imgs_path = path_to_anns
    
    # Get annotated images and segmentation masks from json with polygon annotations
    if os.path.exists(path_to_json):
        parse_annotations(
            path_to_json, 
            path_to_imgs, 
            ann_imgs_path, 
            diversify_string=diversify_string
        )

        # Crop images and annotations in vertical patches
        crop_images(path_to_imgs, path_to_crops, ann_imgs_path, crop_size=crop_size, stride=stride)
        
    else: # Images are copied as annotations to keep the same folder structure of CAUSE
        crop_images(path_to_imgs, path_to_crops, None, crop_size=crop_size,stride=stride)

    # Create the folder structure for the images and annotations complying with the CAUSE dataset structure
    path_tree_generator(dataset_name)  

    # Reorganize images and annotations in train val folders
    split_crops(path_to_crops, dataset_name, train_size=train_size)

    # Remove crops
    shutil.rmtree(path_to_crops)

    if path_to_anns is None:
        shutil.rmtree(ann_imgs_path)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
                    "--path_to_json", type=str, default='result.json', 
                    help="path to json file with annotations in polygon format."
    )

    parser.add_argument(
                    "--path_to_imgs", type=str, default='images', 
                    help="path to folder with images."
    )

    parser.add_argument(
                    "--path_to_anns", default='annotations', 
                    help="path to folder with annotations. If None the annotations \
                          folder is automatically created and then deleted (lightwieght)."
    )

    parser.add_argument(
                    "--dataset_name", type=str, default='dataset', 
                    help="path to folder where to save annotated images."
    )

    parser.add_argument(
                    "--crop_size", default=500, type = int,
                    help="size of the cropped images, accepted int or tuple (height, width)."
    )

    parser.add_argument(
                    "--diversify_string",
                    action="store_true",
                    help="If present, then the class 'Writing normale' will be different from 'Writing zigrinato'; otherwise, they will be the same."
    )
    parser.add_argument(
                    "--reformat_imgs", default=False,
                    help="if true the images names are changed to the format id_nameimg.png \
                          note that this is necessary if your images are not like this"
    )

    parser.add_argument("--stride", default= False,
                        help="if true when cropping the images we use a stride of 250px."
    )

    parser.add_argument("--train_size", default=0.6, type=float,
                        help="train size for the split of the dataset."
    )
    
    args = parser.parse_args()

    main(
        args.path_to_json, # annotations file
        args.path_to_imgs, # images folder
        args.path_to_anns, # segmentation mask folder
        args.dataset_name, # folder where to save images 
        args.crop_size,
        args.diversify_string,
        args.reformat_imgs,
        args.stride,
        args.train_size
    )