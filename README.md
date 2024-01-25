# Pirelli Tyre Semantic Segmentation with CAUSE
This repository contains the implementation of semi-unsupervised semantic segmentation on Pirelli tyres using the [CAUSE(Causal-Unsupervised-Segmentation
)](https://github.com/byungkwanlee/causal-unsupervised-segmentation) framework [1].

 The primary objective of this work was to segment different types of text written on Pirelli tyres, given the challenge of having limited labeled images and a substantial amount of unlabeled images.

## CAUSE Framework
CAUSE is a powerful repository designed for semi-unsupervised semantic segmentation tasks. In our specific case, it played a crucial role in leveraging both the limited supervised images and the abundance of unsupervised images. We conducted experiments using CAUSE and compared the results with two baseline models to showcase the effectiveness of CAUSE when dealing with a scarcity of labeled data.

## Baseline Models
To illustrate the superiority of CAUSE, we trained two baseline models using the available labeled data. These baselines serve as a benchmark for comparison, emphasizing the improved performance achieved by CAUSE using unsupervised images.

## Modifications to CAUSE
In our pursuit of optimizing the segmentation task, we made specific modifications to the CAUSE framework. These adjustments aimed to enhance the model's ability to generate accurate predictions, especially when dealing with the unique characteristics of Pirelli tyre images. Additionally, we explored the use of a Visual Transformer as a backbone instead of DINO to evaluate its impact on performance.

## Image Preprocessing
The repository includes a dedicated file for preprocessing images. Users can customize the preprocessing using various parameters:
- path_to_json: Path to the JSON file containing pixel labels.
- path_to_imgs: Path to the images.
- path_to_anns: Path to the folder with annotations.
- dataset_name: Path to the folder to save annotated images.
- crop_size: Size of the cropped images.
- diversify_string: Option to have 3 or 4 classes for segmentation.
- reformat_imgs: Reformat the images into PNG files.
- stride: If used, creates a stride of 250 pixels.
- train_size: Size to split the dataset.

These parameters provide flexibility for adapting the preprocessing step to different datasets and segmentation requirements

## References
[1] **Causal Unsupervised Semantic Segmentation**
> 
> *Authors: Junho Kim, Byung-Kwan Lee, Yong Man Ro*
> 
> *Published: October 11, 2023*
> 
> [GitHub Repository](https://github.com/username/cause-repository)

# Authors
  - [Kilian Tiziano Le Creurer](https://github.com/Kilian36) 
  - [Jacopo D'Abramo](https://github.com/jacopodabramo)
  - [Lorenzo Cassano](https://github.com/LorenzoCassano) 