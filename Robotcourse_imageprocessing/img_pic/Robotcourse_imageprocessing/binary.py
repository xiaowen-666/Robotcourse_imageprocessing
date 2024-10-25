import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def img_gray(img_path: str):
    """
    :灰度图像处理公式如下
    :Gray = Red*0.299 + Green*0.587 + Blue*0.114
    :函数实现将彩色图像进行灰度化处理, img_path 为彩色图像路径
    :return grayImage, grayChanne:
    """

    img = Image.open(img_path)
    imgData = np.array(img)
    redChannel = imgData[:, :, 0]
    greenChannel = imgData[:, :, 1]
    blueChannel = imgData[:, :, 2]
    grayChannel = (redChannel * 0.299 + greenChannel * 0.587 + blueChannel * 0.114).astype(np.uint8)
    grayImage = np.stack((grayChannel,) * 3, axis=-1)
    return grayImage, grayChannel, imgData


def img_binary_global(img_path, threshold: int):
    """
    :全局阈值处理，img_path 为彩色图像路径， threshold为阈值
    :return None:
    """
    grayImage, grayChannel, imgData = img_gray(img_path)
    binaryImage_global = np.where((grayChannel > threshold) * 255, 0, 255).astype(np.uint8)
    return binaryImage_global


def img_binary_local(img_path, window_size=15, k=0.2):
    """
    :局部阈值处理
    :定义Niblack阈值处理函数
    :return None:
    """
    grayImage, grayChannel, imgData = img_gray(img_path)
    half_window = window_size // 2
    rows, cols = grayChannel.shape
    binary_image = np.zeros_like(grayChannel)

    for r in range(half_window, rows - half_window):
        for c in range(half_window, cols - half_window):
            window = grayChannel[r - half_window:r + half_window + 1, c - half_window:c + half_window + 1]
            mean = np.mean(window)
            std_dev = np.std(window)
            threshold = mean + k * std_dev
            binary_image[r, c] = 255 if grayChannel[r, c] > threshold else 0

    return binary_image


if __name__ == '__main__':
    grayImage, grayChannel, imgData = img_gray('img_pic/rose.jpg')
    binaryImage_global = img_binary_global('img_pic/rose.jpg', 180)
    binaryImage_local = img_binary_local('img_pic/rose.jpg', window_size=11, k=0.2)
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(imgData)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title('Gray Image')
    plt.imshow(grayImage)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title('Binary Image')
    plt.imshow(binaryImage_local, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title('Gray Level Histogram')
    plt.hist(grayChannel.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    plt.xlabel('灰度值')
    plt.ylabel('分布频率')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()
