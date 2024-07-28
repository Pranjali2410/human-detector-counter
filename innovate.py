import cv2
import numpy as np
import os

class HumanDetector:
    def __init__(self, weights_path, cfg_path, names_path):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_and_count_humans(self, frame):
        height, width, _ = frame.shape

        # Detecting objects
        # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Information to display on screen
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # 0 is the class ID for 'person'
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        human_count = len(indexes)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, (0, 255, 0), 3)

        cv2.putText(frame, f"Human Count: {human_count}", (10, 50), font, 3, (0, 0, 255), 3)

        return frame, human_count

def main():
    # File paths
    weights_path = "yolov3.weights"
    cfg_path = "yolov3.cfg"
    names_path = "coco.names"

    # Check if files exist
    for file_path in [weights_path, cfg_path, names_path]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return

    # Initialize detector
    detector = HumanDetector(weights_path, cfg_path, names_path)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame, count = detector.detect_and_count_humans(frame)

        cv2.imshow("Human Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()