import xml.etree.ElementTree as ET
import os
import json

def xml_to_json(input_file, output_file):
    file_dir = []
    root_dir = ""                  # 遍历每幅图像对应的xml文件，并存储名称和根目录
    for rot, dir, file in os.walk(input_file):
        root_dir = rot
        file_dir = file

    images = []
    annotations = []
    categories = []
    count = 0
    object_names = []
    names = []
    id = 0
    for file in file_dir:

        xml_file = open(root_dir + "/" + file, encoding='gbk')
        tree = ET.parse(xml_file)
        # print(root_dir + "/" + file)
        root = tree.getroot()
        img = {}
        img.fromkeys(("id", "file_name", "height", "width"))
        img["id"] = count

        for filename in root.findall('filename'):
            img["file_name"] = filename.text
        for size in root.findall("size"):
            img["height"] = int(float(size.find("height").text))
            img["width"] = int(float(size.find("width").text))
        for object in root.findall("object"):
            name = object.find("name").text
            names.append(name)
            annotation = {}
            annotation.fromkeys(("image_id", "id", "category_id", "bbox", "area", "segmentation", "iscrowd"))
            annotation["image_id"] = count
            annotation["iscrowd"] = 0
            annotation["id"] = id
            annotation["segmentation"] = [[]]
            bbox = object.find("bndbox")
            x_min = int(float(bbox.find("xmin").text))
            x_max = int(float(bbox.find("xmax").text))
            y_min = int(float(bbox.find("ymin").text))
            y_max = int(float(bbox.find("ymax").text))
            box = [x_min, y_min, x_max-x_min, y_max-y_min]
            area = int((x_max-x_min) * (y_max-y_min))
            annotation["area"] = area
            annotation["bbox"] = box
            annotation["category_id"] = name
            annotations.append(annotation)
            id += 1
        count += 1
        images.append(img)
    for name in names:
        if name not in object_names:
            object_names.append(name)
    num = 0
    for object_name in object_names:
        category = {}
        category.fromkeys(("id", "name"))
        category["id"] = num
        category["name"] = object_name
        categories.append(category)
        for annotation in annotations:
            if annotation["category_id"] == object_name:
                annotation["category_id"] = num
        num += 1
    print(categories)
    coco_json = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(coco_json, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行
if __name__=="__main__":
    input_file = "C:/Users/gst01/VOCtrainval/val_xml"
    output_file = "C:\\Users\\gst01\\VOCtrainval\\VOC_val.json"
    xml_to_json(input_file, output_file)
