# Object-Detection-using OpenCV

This repository demonstrates object detection using OpenCV with a pre-trained SSD MobileNet V3 model on the COCO dataset. It includes examples for detecting objects in images and videos.

## Requirements

- Python 3.x
- OpenCV (`cv2` library)
- Matplotlib (for displaying images)
- Pre-trained SSD MobileNet V3 model files:
  - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: Model configuration file
  - `frozen_inference_graph.pb`: Frozen model file
- Class labels file (`labels.txt`): Contains class names corresponding to the model's output classes

## Setup

1. **Install Python Dependencies**

   ```bash
   pip install opencv-python matplotlib
   ```

2. **Download Pre-trained Model Files**

   - Download `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` and `frozen_inference_graph.pb` from [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
   
3. **Prepare Class Labels**

   - Create a `labels.txt` file and populate it with the COCO class names corresponding to the model's output classes.

4. **Usage**

   - **Object Detection in Images:**

     ```python
     import cv2
     import matplotlib.pyplot as plt

     # Load model and class labels
     config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
     frozen_model = 'frozen_inference_graph.pb'
     model = cv2.dnn_DetectionModel(frozen_model, config_file)
     
     class_labels = []
     with open('labels.txt', 'rt') as f:
         class_labels = f.read().rstrip('\n').split('\n')

     # Load and detect objects in an image
     img = cv2.imread('input_image.jpg')
     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

     model.setInputSize(320, 320)
     model.setInputScale(1.0 / 127.5)
     model.setInputMean((127.5, 127.5, 127.5))
     model.setInputSwapRB(True)

     ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
     
     font_scale = 3
     font = cv2.FONT_HERSHEY_PLAIN
     for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
         cv2.rectangle(img, boxes, (255, 0, 0), 2)
         cv2.putText(img, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
     
     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
     plt.show()
     ```

   - **Object Detection in Videos:**

     ```python
     import cv2
     cap = cv2.VideoCapture('video_file.mp4')
     if not cap.isOpened():
         cap = cv2.VideoCapture(0)  # Use primary camera if video file not specified
     if not cap.isOpened():
         raise IOError('Cannot open video capture device')

     font_scale = 3
     font = cv2.FONT_HERSHEY_PLAIN

     while True:
         ret, frame = cap.read()
         if not ret:
             break

         ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

         for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
             if ClassInd <= 80:  # Assuming 80 classes (adjust as per your model)
                 cv2.rectangle(img, boxes, (255, 0, 0), 2)
                 cv2.putText(frame, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

         cv2.imshow('Object Detection', frame)

         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

     cap.release()
     cv2.destroyAllWindows()
     ```
     - **Object Detection in Webcam:**
     ```cap=cv2.VideoCapture(1)
if not cap.isOpened():
     cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Can not open video')

font_scale =3
font = cv2.FONT_HERSHEY_PLAIN

while True:
 ret, frame =cap.read()
    
 ClassIndex, confidece, bbox =model.detect(frame, confThreshold=0.55)

 print(ClassIndex)

 if(len(ClassIndex)!=0):
  for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
          if(ClassInd<=80):
            cv2.rectangle(img, boxes,(255,0,0),2)
            cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font , fontScale=font_scale,color=(0,255,0), thickness=3)
 cv2.imshow('webcam',frame)

 if cv2.waitKey(2) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyaLLWindows() ```

# Load model and class labels (same setup as in previous sections)

cap = cv2.VideoCapture(1)  # Try to open secondary camera
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Try to open primary camera
if not cap.isOpened():
    raise IOError('Cannot open video capture device')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if ClassInd <= 80:  # Assuming 80 classes (adjust as per your model)
            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
            cv2.putText(frame, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


## Notes

- Adjust `confThreshold` and other parameters as per your requirement.
- Ensure your environment has the necessary dependencies installed and the model files are accessible.
- This code assumes you have correctly set up the model files (`pbtxt` and `pb`), as well as the class labels file (`labels.txt`).

---

This README provides an overview of how to set up and use the object detection script with SSD MobileNet V3 using OpenCV. Adjustments may be necessary based on specific model configurations and environment setups.
