import cv2
import numpy as np
from matplotlib import pyplot as plt


def template_matching(target_path, template_path):
    target_image = cv2.imread(target_path)
    target_image= cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    template_image = cv2.imread(template_path)
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
    object_image = target_image.copy()
    target_imgData = np.array(object_image)
    target_grayChannel = (target_imgData[:, :, 0] * 0.299 + target_imgData[:, :, 1] * 0.587 +
                   target_imgData[:, :, 2] * 0.114).astype(np.uint8)
    target_grayImage = np.stack((target_grayChannel,) * 3, axis=-1)
    template_copy = template_image.copy()
    template_imgData = np.array(template_copy)
    template_grayChannel = (template_imgData[:, :, 0] * 0.299 + template_imgData[:, :, 1] * 0.587 +
                            template_imgData[:, :, 2] * 0.114).astype(np.uint8)
    template_grayImage = np.stack((template_grayChannel,) * 3, axis=-1)
    template_h, template_w = template_grayImage.shape[:2]
    result = cv2.matchTemplate(target_grayImage, template_grayImage, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    cv2.rectangle(object_image, top_left, bottom_right, (0, 255, 0), 2)

    return target_image, template_image, object_image



if __name__ == '__main__':
    target_image, template_image, object_image = template_matching("./img_pic/woman_target.png",
                                                                  "img_pic/woman_template.png")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title('Target Image')
    plt.imshow(target_image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Template Image')
    plt.imshow(template_image)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('object Image')
    plt.imshow(object_image)
    plt.axis('off')
    plt.show()
