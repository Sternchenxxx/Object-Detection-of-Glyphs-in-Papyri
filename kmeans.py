import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans

json_path = r"dataset\2a.Training\HomerCompTraining.json"
image_dir = r"new_dataset\images"
output_image_dir = r"splitted_dataset\images"
output_label_dir = r"splitted_dataset\labels"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

category_ids = sorted(set(ann["category_id"] for ann in data["annotations"]))
category_mapping = {old_id: idx for idx, old_id in enumerate(category_ids)}
print("category mapping:", category_mapping)

image_info = {img["id"]: os.path.basename(img["file_name"]) for img in data["images"]}

for image_id, file_name in image_info.items():
    image_path = os.path.join(image_dir, file_name)
    if not os.path.exists(image_path):
        print(f"image does not exist: {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"read failed: {image_path}")
        continue

    print(f"success: {image_path}")

    img_height, img_width, _ = img.shape
    letter_boxes = []
    letter_classes = []

    for ann in data["annotations"]:
        if ann["image_id"] == image_id:
            x_min, y_min, box_w, box_h = ann["bbox"]
            x_center = x_min + box_w / 2
            y_center = y_min + box_h / 2
            letter_boxes.append((x_center, y_center, x_min, y_min, box_w, box_h))
            letter_classes.append(ann["category_id"])

    if len(letter_boxes) < 5:
        print(f"image {file_name} too few letters, skipping")
        continue

    letter_boxes = np.array(letter_boxes)
    xy_centers = letter_boxes[:, :2]
    n_clusters = max(2, min(len(letter_boxes)//10,10))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(xy_centers)

    sorted_rows = {i: [] for i in range(n_clusters)}
    for idx, cluster in enumerate(labels):
        sorted_rows[cluster].append((letter_boxes[idx], letter_classes[idx]))

    for row_idx, row in sorted_rows.items():
        row = sorted(row, key=lambda b: b[0][0])
        if len(row) == 0:
            continue

        for i in range(0, len(row), 10):
            line_boxes = row[i:i + 10]
            if len(line_boxes) == 0:
                continue

            x_min=min(b[0][2]for b in line_boxes)
            y_min=min(b[0][3]for b in line_boxes)
            x_max=max(b[0][2]+ b[0][4] for b in line_boxes)
            y_max=max(b[0][3]+ b[0][5] for b in line_boxes)

            if x_max <= x_min or y_max <= y_min:
                print(f"Invalid cropping region: {file_name}, x: ({x_min}, {x_max}), y: ({y_min}, {y_max})")
                continue

            cropped_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            if cropped_img.size == 0:
                print(f"Skip empty clipping: {file_name} row {row_idx}")
                continue

            line_file_name= f"{os.path.splitext(file_name)[0]}_line{row_idx}_{i // 10}.jpg"
            line_img_path=os.path.join(output_image_dir, line_file_name)
            line_img_path = os.path.normpath(line_img_path)
            os.makedirs(os.path.dirname(line_img_path), exist_ok=True)

            print(f"Try saving the line graph: {line_img_path}")
            success = cv2.imwrite(line_img_path, cropped_img)
            if not success:
                print(f"Unable to save image: {line_img_path}")
                continue

            label_file_path = os.path.join(output_label_dir, line_file_name.replace(".jpg", ".txt"))
            os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
            with open(label_file_path, "w") as f:
                for box, old_class_id in line_boxes:
                    if old_class_id not in category_mapping:
                        print(f"Category not found {old_class_id}，skip")
                        continue

                    new_class_id = category_mapping[old_class_id]
                    abs_x_center, abs_y_center, abs_x_min, abs_y_min, box_w, box_h = box
                    local_x_center = abs_x_center - x_min
                    local_y_center = abs_y_center - y_min
                    local_x_min = abs_x_min - x_min
                    local_y_min = abs_y_min - y_min
                    cropped_width = x_max - x_min
                    cropped_height = y_max - y_min

                    if cropped_width <= 0 or cropped_height <= 0:
                        print(f"Skip invalid box: {file_name} ({local_x_min}, {local_y_min})")
                        continue

                    norm_x_center = local_x_center / cropped_width
                    norm_y_center = local_y_center / cropped_height
                    norm_w = box_w / cropped_width
                    norm_h = box_h / cropped_height
                    norm_x_center = max(0, min(1, norm_x_center))
                    norm_y_center = max(0, min(1, norm_y_center))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))

                    print(f"{file_name} - category: {old_class_id} -> {new_class_id}")
                    f.write(f"{new_class_id} {norm_x_center} {norm_y_center} {norm_w} {norm_h}\n")
print("finished")
