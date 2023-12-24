from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("C:/Users/User/Documents/best.pt")

classNames = ["olive", "olive_ob", "olive_unfocused", "olive_unfocused_ob"]

olive_object_width = 3
olive_d = 50
physical_width = 3
scale_factor = 100

img = cv2.imread("C:/Users/User/Documents/olive_tree2.jpg")
results = model(img)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        class_index = int(box.cls[0])

        # Check if the detected class is an olive
        if classNames[class_index] == "olive":
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            olive_focal_length = (olive_object_width * olive_d) / w
            olive_apparent_width = w  # Apparent width of the olive in pixels
            olive_distance = ((olive_object_width * physical_width) / olive_apparent_width) * scale_factor
            cvzone.putTextRect(img, f'Olive, distance: {int(olive_distance)}cm', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            print(f"Olive, distance {int(olive_distance)}cm")

        else:
            cls = int(box.cls[0])
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

cv2.imshow("Image", img)
cv2.waitKey(1)
cv2.destroyWindow("Image")
