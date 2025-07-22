import os
import re
import numpy as np
import cv2
from tqdm import tqdm
from doclayout_yolo import YOLOv10  # 假设你的模型封装在这个类中

# 初始化模型
model = YOLOv10("/data2/mwk/models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt")  # 替换为你实际模型路径

# 提取 bbox_y 和 bbox_x
def extract_bboxes(dets):
    bbox_y = []
    bbox_x = []
    for det in dets:
        x1, y1, x2, y2 = map(int, det['bbox'])
        bbox_y.append((y1, y2))
        bbox_x.append((x1, x2))
    return bbox_y, bbox_x

# 合并连续坐标区间
def merge_ranges(coords):
    if len(coords) == 0:
        return []
    ranges = []
    start = prev = coords[0]
    for x in coords[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ranges

# 获取未覆盖的 Y/X 区间
def get_uncovered_ranges(size, bboxes, axis=0):
    coverage = np.zeros(size, dtype=bool)
    for a1, a2 in bboxes:
        a1 = max(0, a1)
        a2 = min(size - 1, a2)
        coverage[a1:a2+1] = True
    uncovered = np.where(coverage == False)[0]
    return merge_ranges(uncovered)

# 裁剪图像
def crop_blank_regions(image, blank_y, blank_x):
    cut_y = 0
    for y1, y2 in blank_y:
        y1_cut = max(0, y1 - cut_y)
        y2_cut = max(0, y2 - cut_y)
        image = np.concatenate((image[:y1_cut, :], image[y2_cut+1:, :]), axis=0)
        cut_y += (y2 - y1 + 1)

    cut_x = 0
    for x1, x2 in blank_x:
        x1_cut = max(0, x1 - cut_x)
        x2_cut = max(0, x2 - cut_x)
        image = np.concatenate((image[:, :x1_cut], image[:, x2_cut+1:]), axis=1)
        cut_x += (x2 - x1 + 1)

    return image

# 主函数，处理整个文件夹
def process_folder(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_exts = ['.jpg', '.png', '.jpeg']
    image_list = [f for f in os.listdir(image_dir) if os.path.splitext(f)[-1].lower() in image_exts]

    for image_name in tqdm(image_list, desc="Processing"):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[警告] 无法读取图像: {image_path}")
            continue
        h, w = image.shape[:2]

        # 模型预测
        results = model.predict(image_path, imgsz=1024, conf=0.2, device="cuda:1")  # 根据实际情况调整参数
        # annotated_frame = results[0].plot(pil=True, line_width=5, font_size=20)
        # cv2.imwrite(os.path.join(output_dir,image_name.replace(".png","_new.png")), annotated_frame)
        predictions = results[0]

        # 获取边界框坐标 (xyxy格式)
        boxes = predictions.boxes.xyxy  # 格式: [x1, y1, x2, y2]，分别为左上角和右下角坐标

        # 获取类别标签
        classes = predictions.boxes.cls

        # 获取置信度分数
        confidences = predictions.boxes.conf
        dets = []
        # 打印bbox信息
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cls_id = int(cls.item())
            conf_score = float(conf.item())
            dets.append({"bbox": (x1, y1, x2, y2), "class": cls_id, "conf": conf_score})

        bbox_y, bbox_x = extract_bboxes(dets)
        blank_y_ranges = get_uncovered_ranges(h, bbox_y, axis=0)
        blank_x_ranges = get_uncovered_ranges(w, bbox_x, axis=1)

        # 裁剪图像
        cropped_image = crop_blank_regions(image, blank_y_ranges, blank_x_ranges)

        # 保存结果
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, cropped_image)

    print(f"✅ 所有图像处理完成，结果保存在: {output_dir}")

# 示例调用
if __name__ == "__main__":
    image_folder = "/data2/mwk/datasets/data/docvqa/test/documents"         # 原始图像目录
    output_folder = "output_cropped"  # 输出图像目录
    process_folder(image_folder, output_folder)
