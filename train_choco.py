#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import skimage.draw
from skimage import io
import argparse
import cv2 as cv
import imgaug as ia
import matplotlib.pyplot as plt
# from orb.orb_matcher import ORB_Matcher

import logging
from mmrcnn.config import Config
from mmrcnn import model as model_lib, utils
import json

logging.getLogger('PIL').setLevel(logging.WARNING)
cf = open('./config.json')
json_config = json.load(cf)

class FMCGConfig(Config):
    NAME = 'mobile-choco'
    IMAGES_PER_GPU = int(json_config['images_per_gpu'])
    NUM_CLASSES = len(json_config['classes']) + 1     # background + no distinct products
    STEPS_PER_EPOCH = int(json_config['steps_per_epoch'])
    DETECTION_MIN_CONFIDENCE = float(json_config['detection_min_confidence'])
    BACKBONE = json_config['backbone']


class FMCGDataset(utils.Dataset):

    def load_data(self, dataset_dir, subset, classes):
        # TODO: add support for multiple classes
        for index, source in enumerate(classes):
            self.add_class(source=source, class_id=index+1, class_name=source)

            assert subset in ['train', 'val']

            src_dir = os.path.join(dataset_dir, source)
            _, dirs, _ = next(os.walk(src_dir))
            src_sub_dirs = [dir for dir in dirs if subset in dir]

            for dir in src_sub_dirs:
                # Note: in VIA 2.0, regions was changed from a list to a dict
                annotations = json.load(open(os.path.join(os.path.join(src_dir, dir), 'via_region_data.json')))
                annotations = list(annotations.values())

                annotations = [a for a in annotations if a['regions']]

                for annotation in annotations:
                    if type(annotation['regions']) is dict:
                        polygons = [region['shape_attributes'] for region in annotation['regions'].values()]
                        class_ids = [(classes.index(region['region_attributes']['class'])+1) for region in annotation['regions'].values()]
                    else:
                        polygons = [region['shape_attributes'] for region in annotation['regions']]
                        class_ids = [(classes.index(region['region_attributes']['class'])+1) for region in annotation['regions']]

                    image_path = os.path.join(os.path.join(src_dir, dir), annotation['filename'])
                    image = io.imread(image_path)
                    # TODO: modify the json file to have image shapes within
                    height, width = image.shape[:2]

                    self.add_image(source=source,
                    image_id=annotation['filename'],
                    path=image_path,
                    width = width, height = height,
                    polygons=polygons, class_ids = class_ids)


    def load_mask(self, image_id):
        image_info = self.image_info[image_id]

        if image_info['source'] not in json_config['classes']:
            return super(self.__class__, self).load_mask(image_id)

        mask = np.zeros([image_info['height'], image_info['width'],
                        len(image_info['polygons'])], dtype=np.uint8)

        for index, polygon in enumerate(image_info['polygons']):
            rr, cc = skimage.draw.polygon(polygon['all_points_y'], polygon['all_points_x'])

            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1

            mask[rr, cc, index] = 1

        # returns mask and array of class IDs of each instance
        return mask.astype(np.bool), np.array(image_info['class_ids'])


    def image_reference(self, image_id):
        if self.image_info[image_id]['source'] not in json_config['classes']:
            return self.image_info[image_id]['path']
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    train_dataset = FMCGDataset()
    print('q')
    train_dataset.load_data(args.dataset, 'train', classes=json_config['classes'])
    print('w')
    train_dataset.prepare()
    print('e')

    val_dataset = FMCGDataset()
    val_dataset.load_data(args.dataset, 'val', classes=json_config['classes'])
    val_dataset.prepare()

    iaa = ia.augmenters
    some = iaa.Sometimes(2/3, iaa.OneOf([
        iaa.Flipud(1.0),
        # iaa.Crop(percent=(0, 0.1)),
        iaa.GaussianBlur(sigma=(0, 1.5)),
        iaa.Multiply((0.5, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.2*255), per_channel=0.5),
        iaa.Affine(
            scale={"x": (0.5, 1.2), "y": (0.5, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-8, 8)
        )
    ]))

    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE,
                epochs=int(args.epochs),
                augmentation=some,
                layers='heads')


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)

    return splash




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect FMCG')

    parser.add_argument('--epochs', default=json_config['epochs'], help='Directory of the dataset')

    parser.add_argument('--dataset', default=json_config['dataset_dir'], help='Directory of the dataset')

    parser.add_argument('--weights', default='coco', help="Path to weights .h5 file or 'coco'")

    parser.add_argument('--logs', default=json_config['logs_path'], help='Logs and checkpoints directory')

    args = parser.parse_args()

    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Logs: ', args.logs)

    config = FMCGConfig()
    config.display()


    model = model_lib.MaskRCNN(mode='training', config=config, model_dir=args.logs)
    if args.weights.lower() == 'last':
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    print('Loading weights... ', weights_path)

    if args.weights.lower() == 'coco':
        model.load_weights(json_config['coco_weights_path'], by_name=True,
        exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
    else:
        model.load_weights(weights_path, by_name=True)
    train(model)