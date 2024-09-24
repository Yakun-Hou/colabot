import cv2
import numpy as np
import cv2
import numpy as np
import os

def fuse_rgb_depth(rgb_path, depth_path, save_path):
    # Load RGB and depth images
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    # print(depth_image)
    # 将RGB图像和深度图像叠加
    fused_image = cv2.addWeighted(rgb_image, 0.5, depth_image, 0.5, 0)

    # 对融合后的图像进行卷积操作
    kernel = np.ones((5, 5), np.float32) / 25
    fused_image = cv2.filter2D(fused_image, -1, kernel)

    # Save fused image to disk
    filename = os.path.basename(rgb_path)
    save_path = os.path.join(save_path, filename)
    cv2.imwrite(save_path, fused_image)
    
    return fused_image

# 设置RGB图像和深度图像的路径
rgb_dir = "/media/julien/TOSHIBA EXT/yolo-data/dev/train/001/img"
depth_dir = "/media/julien/TOSHIBA EXT/yolo-data/dev/train/001/d"

# 设置保存融合图像的路径
save_dir = "/media/julien/TOSHIBA EXT/yolo-data/dev/train/001/fused_img"

# 遍历所有的RGB图像和深度图像，并进行融合和保存
for i in range(200):
    rgb_path = os.path.join(rgb_dir, "{:04d}.png".format(i))
    depth_path = os.path.join(depth_dir, "{:04d}.tif".format(i))
    save_path = os.path.join(save_dir, "{:04d}.jpg".format(i))
    fuse_rgb_depth(rgb_path, depth_path, save_dir)


