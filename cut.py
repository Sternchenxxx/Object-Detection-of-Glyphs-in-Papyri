import os
import cv2
import numpy as np
from ultralytics import YOLO

input_dir = r"dataset_1\GRK-papyri\All\All"
model_path = r"output4\papyri_train\weights\best.pt"
output_dir = r"results\detected_full"
vis_dir = os.path.join(output_dir, "visualization")
patch_save = os.path.join(output_dir, "patches")

#window parameters
patch_size =256
stride=128
conf_thresh=0.3
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(patch_save, exist_ok=True)
model=YOLO(model_path)
colors=[(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in range(20)]

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(root, file)
        gray=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        vis_img = rgb.copy()
        h, w = gray.shape
        base_name = os.path.splitext(file)[0]

        #Cut/detect the image in sliding window
        for y in range(0,h-patch_size+1,stride):
            for x in range(0,w-patch_size+1,stride):
                patch= rgb[y:y+patch_size,x:x+patch_size]
                result=model(patch,conf=conf_thresh)[0]
                boxes=result.boxes

                if boxes is not None and boxes.shape[0] > 0:
                    #Save the patch containing the characters
                    patch_name = f"{base_name}_{y}_{x}.png"
                    cv2.imwrite(os.path.join(patch_save, patch_name), gray[y:y + patch_size, x:x + patch_size])

                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        #Map coordinates back to original image
                        x1,y1,x2,y2=xyxy
                        abs_x1=x1+x
                        abs_y1=y1+y
                        abs_x2=x2+x
                        abs_y2=y2+y
                        cv2.rectangle(vis_img, (abs_x1, abs_y1), (abs_x2, abs_y2), colors[cls_id % len(colors)], 2)
                        cv2.putText(vis_img, f"{cls_id}", (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[cls_id % len(colors)], 1)
        cv2.imwrite(os.path.join(vis_dir, base_name + ".jpg"), vis_img)
        print(f"{file} processed")