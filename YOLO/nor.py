import os

label_dir = "/home/julien/annotation/"
image_width = 768
image_height = 432

# 遍历所有的标签文件，并将类别移动到最前面，然后归一化边界框坐标, 去掉跟踪有关标签
for filename in os.listdir(label_dir):
    label_path = os.path.join(label_dir, filename)
    with open(label_path, "r") as f:
        lines = f.readlines()
        a=[]
        for i, line in enumerate(lines):
            line = line.strip().split()
            class_id = int(line[-1])
            if class_id!= 99:
                x1, y1, x2, y2 = map(float, line[:-1])
                x = (x1 + x2) / 2 / image_width
                y = (y1 + y2) / 2 / image_height
                w = abs((x2 - x1))/ image_width
                h = abs((y2 - y1))/ image_height
                if abs(x1-x2)<40 or abs(y1-y2)<40: 
                    continue
                else:
                    a.append("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, x, y, w, h))
    with open("/home/julien/new/"+filename, "w") as f:
            f.writelines(a)
# import os


# label_dir = "/home/julien/yolo-dataset/002/labels/train/"
# image_width = 768
# image_height = 432

# # 遍历所有的标签文件，并将类别移动到最前面，然后归一化边界框坐标
# for filename in os.listdir(label_dir):
#     label_path = os.path.join(label_dir, filename)
#     with open(label_path, "r") as f:
#         lines = f.readlines()
#         for i, line in enumerate(lines):
#             line = line.strip().split()
#             class_id = int(line[0])
#             if class_id!= 0 and class_id!=1:
#                 print(label_path)
#     #         # 将类别移动到最前面
#     #             # lines = [lines[-1]] + lines[:-1]
#     # with open("/home/julien/new/"+filename, "w") as f:
#     #         f.writelines(a)