#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
import cv2
from utils.augmentations import preprocess




class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, img_path, h, w = self.pull_item(index)
        return img, target, img_path

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        
        #img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        return torch.from_numpy(img), target, image_path, im_height, im_width
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes
    


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    paths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        paths.append(sample[2])
    return torch.stack(imgs, 0), targets, paths


def Compute_Darklevel(img, a = 0.06, b = 110, scale = 0.9):

    # img가 torch tensor이면 numpy로 변경
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    # 이미지 범위를 0~255로 설정
    if img.max() <= 1.0:
        img = img * 255

    # YUV 변환 후 Y 채널 추출
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YUV)
        luminance = img_yuv[:, :, 0]
    else:
        luminance = img  # 이미 grayscale이면 그대로 사용 가능

    # 평균 휘도값
    luminance = np.mean(luminance)

    # Sigmoid를 이용해 darklevel 계산
    darklevel = (1.0 / (1.0 + np.exp(-a * (luminance - b)))) * scale

    return torch.tensor(darklevel, dtype=torch.float32)

if __name__ == '__main__':
    from config import cfg
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(14)
