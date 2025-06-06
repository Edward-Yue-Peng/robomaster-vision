import cv2 as cv
from matplotlib import pyplot as plt
import os
from rmvision.processor import process_frame

input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")

img_extension = ('.jpg', '.jpeg', '.png', '.bmp')
img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_extension)]

for img_name in img_files:
    img_path = os.path.join(input_dir, img_name)
    img = cv.imread(img_path)

    result = process_frame(img)[0]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Result')
    plt.suptitle(img_name)
    plt.tight_layout()
    plt.show()
