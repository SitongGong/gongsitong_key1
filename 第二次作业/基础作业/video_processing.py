import cv2
import json
import numpy as np

# 将视频中的各帧保存为图像
def video_to_pic(input_file):
    cap = cv2.VideoCapture(input_file)
    c = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("balloons_gray/balloon"+format(str(c))+".jpg", gray_img)
        cv2.imwrite("balloons_rgb/balloon"+format(str(c))+".jpg", frame)
        c += 1

# 为图像文件夹设置.json标注文件
def coco_dict_json(input_file):
    images = []
    annotations = []
    categories = [{"id": 0, "name": "balloon"}]
    cap = cv2.VideoCapture(input_file)
    c = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        img = {}
        img.fromkeys(("id", "file_name", "height", "width"))
        img["id"] = c
        img["file_name"] = "balloon"+str(c)+".jpg"
        img["height"] = frame.shape[0]
        img["width"] = frame.shape[1]
        annotation = {}
        annotation.fromkeys(("image_id", "id", "category_id", "bbox", "area", "segmentation", "iscrowd"))
        annotation["image_id"] = c
        annotation["id"] = c
        annotation["category_id"] = 0
        annotation["bbox"] = []
        annotation["area"] = frame.shape[0] * frame.shape[1]
        annotation["segmentation"] = [[]]
        annotation["iscrowd"] = 0
        c += 1
        images.append(img)
        annotations.append(annotation)

    coco_json = {"images": images, "annotations": annotations, "categories": categories}
    print(coco_json)
    with open("coco_json.json", "w", encoding='utf-8') as f:
        json.dump(coco_json, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行

# 将带有红色分割标记的图像转换为灰度图像
def rgb_to_gray():

    # H、S、V范围一：
    lower1 = np.array([0, 170, 46])
    upper1 = np.array([10, 255, 255])

    # H、S、V范围二：
    lower2 = np.array([156, 170, 46])
    upper2 = np.array([180, 255, 255])
    count = 0

    for i in range(125):
        img = cv2.imread("balloons/balloon"+str(count)+".jpg")
        image = cv2.imread("balloons_rgb/balloon" + str(count) + ".jpg")
        grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
        mask2 = cv2.inRange(grid_HSV, lower2, upper2)
        mask3 = mask1 + mask2     # 将两个二值图像结果 相加
        img_copy = image[:, :, :]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask3[i][j] == 0:
                    img_copy[i][j][0] = 0.2126 * image[i][j][0] + 0.7152 * image[i][j][1] + 0.0722 * image[i][j][2]
                    img_copy[i][j][1] = img_copy[i][j][0]
                    img_copy[i][j][2] = img_copy[i][j][0]
        cv2.imwrite("balloons_new/"+"balloon"+str(count)+".jpg", img_copy)
        count += 1

def pic_to_video():
    img = cv2.imread('balloons/balloon0.jpg')
    size = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
    videowrite = cv2.VideoWriter('video.avi', fourcc, 25, size)     # 写入对象 1 file name 2 编码器 3 帧率 4 尺寸大小

    for i in range(125):
        img = cv2.imread("balloons_new/balloon" + str(i) + ".jpg")
        videowrite.write(img)

if __name__=="__main__":
    input_file = "test_video.mp4"
    # video_to_pic(input_file)
    # coco_dict_json(input_file)
    # rgb_to_gray()
    pic_to_video()
