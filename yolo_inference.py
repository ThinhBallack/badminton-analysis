from ultralytics import YOLO

model = YOLO('yolov8x.pt')
# model = YOLO('models/best_0307detection_BEST.pt')
# model = YOLO('models/court-keypoints.pt')

result = model('input_videos/DoubleServe.mp4', stream=True)
print(result)
print("boxes:")
for box in result:
    print(box.boxes)