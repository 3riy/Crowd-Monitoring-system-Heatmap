import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN

# Load model
net = cv2.dnn.readNetFromCaffe(
    'mobilenet/MobileNetSSD_deploy.prototxt',
    'mobilenet/MobileNetSSD_deploy.caffemodel'
)

# Classes (focus only on people)
PERSON_CLASS_ID = 15

# Start webcam
cap = cv2.VideoCapture(0)

ALERT_THRESHOLD = 5
CLUSTER_DISTANCE = 75  # Adjust based on camera scale
CLUSTER_SIZE_THRESHOLD = 3  # Number of people per cluster

ret, frame = cap.read()
h, w = frame.shape[:2]
heatmap = np.zeros((h, w), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    people_centroids = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.5 and class_id == PERSON_CLASS_ID:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if 0 <= cx < w and 0 <= cy < h:
                people_centroids.append([cx, cy])
                heatmap[cy, cx] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 3)

    # Cluster detection
    cluster_count = 0
    if len(people_centroids) > 0:
        people_np = np.array(people_centroids)
        clustering = DBSCAN(eps=CLUSTER_DISTANCE, min_samples=CLUSTER_SIZE_THRESHOLD).fit(people_np)
        labels = clustering.labels_

        unique_clusters = set(labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise points
        cluster_count = len(unique_clusters)

    # Heatmap
    heatmap_blur = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))
    overlay = cv2.addWeighted(heatmap_color, 0.6, frame, 0.4, 0)

    # Display crowd info
    cv2.putText(overlay, f"People: {len(people_centroids)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Clusters: {cluster_count}", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Optional alert
    if len(people_centroids) >= ALERT_THRESHOLD:
        cv2.putText(overlay, "ALERT: CROWD FORMING", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display
    cv2.namedWindow("CrowdHawk", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CrowdHawk", 1280, 720)
    cv2.imshow("CrowdHawk", overlay)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
