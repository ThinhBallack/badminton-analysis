from ultralytics import YOLO

model = YOLO('models/court-keypoints.pt')

results = model('input_videos/DoubleServe.mp4', stream=True)

# View results
for r in results:
    print(r.keypoints)  # print the Keypoints object containing the detected keypoints