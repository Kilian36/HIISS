import os 
import json
from pycocotools import mask
import cv2 
import numpy as np
import shutil
import matplotlib.pyplot as plt
from constants import *


def parse_annotations(
                        path_to_json: str, 
                        path_to_imgs: str, 
                        path_to_save: str,
                        diversify_string: bool = True):
    '''
    Creates and save annotated images from json file with annotations in polygon format. Note 
    this is been used with few images and few annotations, and it loads in memory all the images as numpy arrays.
    If you have a lot of images consider to modify this functions to parse the json image by image.

    :param path_to_json: path to json file with annotations in polygon format.
    :param path_to_imgs: path to folder with images.
    :param path_to_save: path to folder where to save annotated images.
    :param diversify_string: if True, then the class "Writing normale" will be
                             different from "Writing zigrinato", otherwise they will
                             be the same.
    
    :return: None 

    '''
    with open(path_to_json, 'r') as file:
        coco_data = json.load(file)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        os.makedirs(os.path.join(path_to_save, "anns_image"))
        os.makedirs(os.path.join(path_to_save, "anns"))

    imgs = {image_path: cv2.imread(os.path.join(path_to_imgs, image_path)) for image_path in sorted(os.listdir(path_to_imgs))}
    anns = {img_name: np.zeros(img.shape[:2]) for img_name,img in imgs.items()} # Zero initialized segmentation arrays (background)

    i = 0
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        segmentation = annotation['segmentation']
        category = annotation["category_id"]

        # Create a unique color for each category
        if category == 0: # "Writing normale"
            class_id = 2
            color = (0, 0, 255)
        elif category == 1: # "Writing zigrinato"

            if diversify_string:
                class_id = 3
                color = (0, 255, 0) # Green
            else: # Writings converge to the same class
                class_id = 2
                color = (0, 0, 255)

        elif category == 2: # "Zigrinato generico"
            class_id = 1
            color = (255, 255, 255) # White

        # Create segmentation mask
        rles = mask.frPyObjects(segmentation, coco_data['images'][image_id]['height'], coco_data['images'][image_id]['width'])
        mask_array = mask.decode(rles).squeeze()
        image_name = coco_data['images'][image_id]['file_name'].split("\\")[-1]

        for img_name,img in anns.items():
            if image_name in img_name:
                anns_img = img
                img_idx = img_name
                break
        mask_ids = np.argwhere((mask_array*class_id) > anns_img)
        mask_idsx = mask_ids[:, 0] 
        mask_idsy = mask_ids[:, 1]

        # Apply the color to the image based on the segmentation mask
        imgs[img_idx][mask_idsx, mask_idsy] = color
        anns[img_idx][mask_idsx, mask_idsy] = class_id
        
        if (i + 1) % 50 == 0:
            print(f"Annotations {i}, image {image_id}")  
        i += 1
        
    
    # Save annotated images
    for img, annot in zip(imgs.items(), anns.items()):
        print(f"I am saving as {img[0].split('_')[0]} the image {img[0]}{annot[0]}")
        cv2.imwrite(os.path.join(path_to_save, "anns_image", f"{img[0].split('_')[0]}_.png"), img[1])
        cv2.imwrite(os.path.join(path_to_save, "anns", f"{annot[0].split('_')[0]}_.png"), annot[1])


def path_tree_generator(path_to_imgs):
    '''
    Generater the folder structure for the images and annotations complying with
    the CAUSE dataset structure.  

    :param path_to_imgs: path to folder with images.

    '''
    img_path   = os.path.join(path_to_imgs,'img') 
    label_path = os.path.join(path_to_imgs,'label') 

    img_train_path  = os.path.join(img_path,'train')
    img_val_path    = os.path.join(img_path,'val')
    img_test_path   = os.path.join(img_path,'test')

    label_train_path = os.path.join(label_path,'train')
    label_val_path   = os.path.join(label_path,'val')
    label_test_path  = os.path.join(label_path,'test')


    curated_path = os.path.join(path_to_imgs,'curated','train')

    if os.path.exists(path_to_imgs):
        shutil.rmtree(path_to_imgs)
    
    os.makedirs(img_train_path)
    os.makedirs(img_val_path)
    os.makedirs(img_test_path)

    os.makedirs(label_train_path)
    os.makedirs(label_val_path)
    os.makedirs(label_test_path)

    os.makedirs(curated_path)


def reformat_imgs(path_to_imgs):
    '''
    Reformat images names to the format id_nameimg.png. Note that this is necessary
    if your images are not like this.

    :param path_to_imgs: path to folder with images.

    '''
    for i, img_name in enumerate(os.listdir(path_to_imgs)):
        new_name = f"{i}_{img_name}"
        os.rename(os.path.join(path_to_imgs, img_name), os.path.join(path_to_imgs, new_name))



def visualize_segmentation_map(segmentation_map_path, class_colors, save_path=None):
    """
    Visualize a segmentation map by assigning colors to different classes and optionally save or display the result.

    :param segmentation_map_path: Path to the grayscale segmentation map image.
    :param class_colors         : Dictionary mapping class labels to RGB colors 
                                  (e.g., {0: [255, 0, 0], 1: [0, 255, 0], ...}).
    :param save_path            : Optional. If provided, save the visualization to the specified path.
                                  Otherwise, display the result.
    """
    # Load the segmentation map in grayscale
    segmentation_map = cv2.imread(segmentation_map_path, cv2.IMREAD_GRAYSCALE)

    # Create an empty RGB image
    height, width = segmentation_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors to different classes
    for label, color in class_colors.items():
        indexes = np.where(segmentation_map == label)
        rgb_image[indexes] = color

    # Display or save the result
    if save_path is not None:
        cv2.imwrite(save_path, rgb_image)
    else:
        plt.imshow(rgb_image)
        plt.title("Segmentation Map Visualization")
        plt.axis("off")
        plt.show()

