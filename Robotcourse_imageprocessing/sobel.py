import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def convertu8(num):
    if num > 255 or num < -255:
        return 255
    elif -255 <= num <= 255:
        if abs(num - int(num)) < 0.5:
            return np.uint8(abs(num))
        else:
            return np.uint8(abs(num)) + 1


def Sobel(img_path, k=0):
    img = cv.imread(img_path)
    row = img.shape[0]
    col = img.shape[1]
    image = np.zeros((row, col), np.uint8)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            y = int(img[i - 1, j + 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                    int(img[i, j + 1, k]) - int(img[i, j - 1, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i + 1, j - 1, k])
            x = int(img[i + 1, j - 1, k]) - int(img[i - 1, j - 1, k]) + 2 * (
                    int(img[i + 1, j, k]) - int(img[i - 1, j, k])) + int(img[i + 1, j + 1, k]) - int(
                img[i - 1, j + 1, k])
            image[i, j] = convertu8(abs(x) * 0.5 + abs(y) * 0.5)
    return image



def Laplacian(img_path):
    img = Image.open(img_path)
    imgData = np.array(img)
    redChannel = imgData[:, :, 0]
    greenChannel = imgData[:, :, 1]
    blueChannel = imgData[:, :, 2]
    grayChannel = (redChannel * 0.299 + greenChannel * 0.587 + blueChannel * 0.114).astype(np.uint8)
    grayImage = np.stack((grayChannel,) * 3, axis=-1)

    # 拉普拉斯算法
    dst = cv.Laplacian(grayImage, cv.CV_16S, ksize=3)
    Laplacian = cv.convertScaleAbs(dst)
    return Laplacian, img


if __name__ == '__main__':
    sobel_img = Sobel("img_pic/12240303_80d87f77a3_n.jpg", 0)
    laplacian_img, img = Laplacian("img_pic/12240303_80d87f77a3_n.jpg")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 2)
    plt.title('Sobel Image')
    plt.imshow(sobel_img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Laplacian Image')
    plt.imshow(laplacian_img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()