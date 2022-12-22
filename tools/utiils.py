import os
from typing import List
import cv2
import numpy as np
from PIL import Image

def train_img_label(save_txt_path:str,train_list:List,valid_list:List) -> None:
    with open(os.path.join(save_txt_path,'train_label.txt'),'a') as f:
        for line in train_list :
            f.write(line +'\t'+' '.join(line.split('_')[:-1])+'\n' )
    with open(os.path.join(save_txt_path,'valid_label.txt'),'a') as f:
        for line in valid_list :
            f.write(line +'\t'+' '.join(line.split('_')[:-1])+'\n' )

def load_dict(path: str) -> List[str]:
    word_dict = []
    with open(path,"r",encoding="utf8") as d:
        word_dict = [l.split('\t') for l in d.read().splitlines() if len(l) > 0]

    return word_dict


def dilate(img_path, save_en_path):
    for file in os.listdir(img_path):
        image = cv2.imread(os.path.join(img_path, file), 3)
        kernel = np.ones((2, 2), np.uint8)
        dilate = cv2.dilate(image, kernel, iterations=1)
        cv2.imwrite(os.path.join(save_en_path, file), dilate)


def opening(img_path, save_en_path):
    for file in os.listdir(img_path):
        image = cv2.imread(os.path.join(img_path, file), 3)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
        cv2.imwrite(os.path.join(save_en_path, file), opening)


def closing(img_path, save_en_path):
    # 1.enrode 2.dilate
    for file in os.listdir(img_path):
        image = cv2.imread(os.path.join(img_path, file), 3)
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
        cv2.imwrite(os.path.join(save_en_path, file), closing)