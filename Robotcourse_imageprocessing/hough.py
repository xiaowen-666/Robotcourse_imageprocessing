import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_lines(img, houghLines, color=[255, 0, 0], thickness=2):
    for line in houghLines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def img_gray(img):
    imgData = np.array(img)
    redChannel = imgData[:, :, 0]
    greenChannel = imgData[:, :, 1]
    blueChannel = imgData[:, :, 2]
    grayChannel = (redChannel * 0.299 + greenChannel * 0.587 + blueChannel * 0.114).astype(np.uint8)
    grayImage = np.stack((grayChannel,) * 3, axis=-1)
    return grayImage


def hough(img_gray, ):
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 50, 120)

    rho = 1
    theta = np.pi / 180
    threshold = 100
    hough_lines = cv2.HoughLines(img_edges, rho, theta, threshold)

    return hough_lines


if __name__ == '__main__':
    img = cv2.imread("img_pic/road.png")
    img_gray = img_gray(img)
    hough_lines = hough(img_gray)
    img_lines = np.zeros_like(img)
    draw_lines(img_lines, hough_lines)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("source", fontsize=12)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_edges, cmap="gray")
    plt.title("edge", fontsize=12)
    plt.axis("off")

    plt.show()
