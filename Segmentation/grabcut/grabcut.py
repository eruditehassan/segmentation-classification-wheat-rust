import numpy as np
import cv2
from PIL import Image
import time


def load_image(img_path):
    image = cv2.imread(img_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD
    return mask

def remove_background(image):
    h, w = image.shape[:2]
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = cv2.bitwise_and(image, image, mask = mask_binary)
    #add_contours(result, mask_binary) # optional, adds visualizations
    return result

def process_image(img_path):
    im = load_image(img_path)
#     for i in range(1,len(train_images)*2, 2):
#         im = cv2.resize(train_images.iloc[i // 2], (1024, 1541))
    im = remove_background(im)
    img = Image.fromarray(im)
    img.save('output/output.jpg')


if __name__ == '__main__':
    process_image('input/input.jpg')
