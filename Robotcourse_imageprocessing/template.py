# # 导入必要的包
# import cv2
#
# # 加载主图像和模板图像
# image = cv2.imread("./img_pic/woman_target.png")
# template = cv2.imread("img_pic/woman_template.png")
#
# # 制作图像的副本
# image_copy = image.copy()
#
# # 将图像转换为灰度图像
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#
# # 获取模板图像的宽度和高度
# print(template.shape)
# template_h, template_w = template.shape[:2]
#
# # 使用归一化交叉相关方法执行模板匹配
# result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
#
# # 找到结果矩阵中最佳匹配的位置
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#
# # 在最佳匹配周围绘制矩形
# top_left = max_loc
# bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
# cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)
#
# # 显示图像
# # cv2.imshow("Image", image)
# # cv2.imshow("Template", template)
# cv2.imshow("Matched Template", image_copy)
# cv2.waitKey(0)



# import cv2
# import numpy as np
#
# # 设置模板匹配和非极大值抑制阈值
# thresh = 0.98
# nms_thresh = 0.6
#
# # 加载主图像和模板图像
# image = cv2.imread("img_pic/woman_target2.png")
# template = cv2.imread("img_pic/woman_template.png")
#
# # 制作图像的副本
# image_copy = image.copy()
#
# # 将图像转换为灰度图像
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#
# # 获取模板图像的宽度和高度
# template_h, template_w = template.shape[:2]
#
# # 使用归一化交叉相关方法执行模板匹配
# result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
#
# # 获取高于阈值的匹配位置的坐标
# y_coords, x_coords = np.where(result >= thresh)
#
# print("找到的匹配数量:", len(x_coords))
#
# # 循环遍历坐标并在匹配周围绘制矩形
# for x, y in zip(x_coords, y_coords):
#     cv2.rectangle(image_copy, (x, y), (x + template_w, y + template_h), (0, 255, 0), 2)
#
# # 显示图像
# cv2.imshow("Template", template)
# cv2.imshow("Multi-Template Matching", image_copy)
# cv2.waitKey(0)
