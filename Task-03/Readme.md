# Transfer Learning with YOLOv3-tiny for Object Detection

### Introduction
In this project, I demonstrate transfer learning using the YOLOv3-tiny model for object detection. It uses a random object detection dataset to evaluate the model's performance and metrics. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system that applies a single neural network to the full image. YOLOv3-tiny is a smaller, faster version of YOLOv3, making it suitable for real-time applications on limited hardware.

### Model Selection
I've chosen the YOLOv3-tiny model for this project due to its efficiency and relatively good accuracy. YOLOv3-tiny is a lightweight version of YOLOv3, making it suitable for applications requiring faster inference times. 

### Dataset 
The COCO (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset. 

## Code Explanation
Loading the Model and Dataset
We start by loading the pre-trained YOLOv3-tiny weights and configuration files. We also load the class names from the COCO dataset.

```py
import cv2
import numpy as np
import matplotlib.pyplot as pyplt

# Load YOLOv3-tiny model
yolo = cv2.dnn.readNet('./yolov3-tiny.weights','./yolov3-tiny.cfg')

# Load class names
classes = []
with open("./coco.names", "r") as f:
    classes = f.read().splitlines()
```
### Preprocessing the Data
We read an image and convert it into a blob, which is the format required by YOLO for processing.
* A blob is a 4-dimensional array (tensor) with dimensions [batch size, number of channels, height, width].
* a blob is used for normalization , resizing and also for changing colour channels.
```py
# Load and preprocess image
img = cv2.imread('./person_cycle.jpg')
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)

# Display the blob
b = blob[0, 0, :, :]
pyplt.imshow(b, cmap='gray')
```

###  Postprocessing + Visualization
We process the model's output to extract bounding boxes, confidences, and class IDs. We then apply non-maximum suppression to filter overlapping boxes and visualize the results.
```py
# Get image dimensions
height, width = img.shape[:2]

# Initialize lists for detected objects
boxes = []
confidences = []
class_ids = []

# Extract bounding boxes, confidences, and class IDs
for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Initialize font and colors for bounding boxes
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Draw bounding boxes and labels on the image
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    conf = str(round(confidences[i], 2))
    color = colors[class_ids[i]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label + " " + conf, (x, y - 10), font, 1, (255, 255, 255), 2)

# Convert image to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Print detections
print(f"Detections: {len(boxes)}")
for i, box in enumerate(boxes):
    print(f"Box {i}: {box}, Confidence: {confidences[i]}, Class ID: {class_ids[i]}")

# Display the final image with detections
pyplt.imshow(img_rgb)
pyplt.show()
```

