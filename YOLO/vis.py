import cv2
import os

# 设置标签文件的路径和图像的路径
label_dir = '/home/julien/yolo-dataset/002/1'
image_dir = "/home/julien/yolo-dataset/002/2"

# 设置图像的宽度和高度
image_width = 768
image_height = 432

# 设置类别标签
class_names = ["person", "car", "truck", "bus"]

# 遍历所有的标签文件，并可视化边界框坐标到图像上
for filename in os.listdir(label_dir):
    label_path = os.path.join(label_dir, filename)
    image_path = os.path.join(image_dir, os.path.splitext(filename)[0] + ".png")
    image = cv2.imread(image_path)
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            class_id = int(line[0])
            x, y, w, h = map(float, line[1:])
            x1 = int((x - w / 2) * image_width)
            y1 = int((y - h / 2) * image_height)
            x2 = int((x + w / 2) * image_width)
            y2 = int((y + h / 2) * image_height)
            class_name = class_names[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # 设置图像的路径
# image_path = "/home/julien/yolo-dataset/002/2/0000.jpg"

# # 设置边界框的绝对坐标和类别
# x1, y1, x2, y2 = 310, 220, 334, 288
# class_id = 0

# # 读取图像
# image = cv2.imread(image_path)

# # 可视化边界框
# cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.putText(image, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # 显示图像
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()