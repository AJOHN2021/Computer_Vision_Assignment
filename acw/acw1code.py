import cv2 
import numpy as np
import argparse
import os

# Define sign classes
SIGN_CLASSES = {
    1: "roundabout", 2: "double bend", 3: "dual carriageway ends",
    4: "traffic lights", 5: "roadworks", 6: "ducks",
    7: "turn left", 8: "keep left", 9: "mini roundabout", 10: "one way",
    11: "warning", 12: "give way", 13: "no entry", 14: "stop",
    15: "20MPH", 16: "30MPH", 17: "40MPH", 18: "50MPH", 19: "national speed limit"
}

# Define color ranges for detection (HSV format)
COLOR_RANGES = {
    "red": [(0, 100, 100), (10, 255, 255)],  # Red
    "blue": [(100, 100, 100), (140, 255, 255)],  # Blue
    "yellow": [(20, 100, 100), (40, 255, 255)]  # Yellow (warning signs)
}

def get_sign_id(sign_name):
    for sign_id, name in SIGN_CLASSES.items():
        if name.lower() == sign_name.lower():
            return sign_id
    return 0  # Unknown

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_shapes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_signs = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)

        if area > 500:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if len(approx) == 3:
                detected_signs.append(("warning", x, y, w, h))
            elif len(approx) == 4 and 0.8 < aspect_ratio < 1.2:
                detected_signs.append(("give way", x, y, w, h))
            elif len(approx) > 5:
                detected_signs.append(("30MPH", x, y, w, h))

    return detected_signs

def detect_signs(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = preprocess_image(image)
    detected_shapes = detect_shapes(edges)

    final_detections = []
    seen_boxes = set()

    for sign, x, y, w, h in detected_shapes:
        box = (x, y, w, h)
        if box in seen_boxes:
            continue  # Avoid duplicates
        seen_boxes.add(box)

        roi = hsv[y:y+h, x:x+w]

        for color, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(roi, np.array(lower), np.array(upper))
            if cv2.countNonZero(mask) > 100:
                final_detections.append((sign, color, x, y, w, h))

    return final_detections

def process_image(image_path, output_file):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load {image_path}")
        return

    filename = os.path.basename(image_path)
    detections = detect_signs(image)
    img_h, img_w = image.shape[:2]

    for sign, color, x, y, w, h in detections:
        sign_id = get_sign_id(sign)
        norm_x = round((x + w / 2) / img_w, 4)
        norm_y = round((y + h / 2) / img_h, 4)
        norm_w = round(w / img_w, 4)
        norm_h = round(h / img_h, 4)
        confidence = 1.0
        with open(output_file, 'a') as f:
            f.write(f"{filename},{sign_id},{sign},{norm_x},{norm_y},{norm_w},{norm_h},0,0,{confidence}\n")

        # Optional: draw bounding box for visualization
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, sign, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Save annotated image
    output_img_path = os.path.splitext(filename)[0] + "_annotated.jpg"
    cv2.imwrite(output_img_path, image)

def process_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        detections = detect_signs(frame)
        img_h, img_w = frame.shape[:2]
        timestamp = round(frame_number / cap.get(cv2.CAP_PROP_FPS), 2)

        for sign, color, x, y, w, h in detections:
            sign_id = get_sign_id(sign)
            norm_x = round((x + w / 2) / img_w, 4)
            norm_y = round((y + h / 2) / img_h, 4)
            norm_w = round(w / img_w, 4)
            norm_h = round(h / img_h, 4)
            confidence = 1.0
            with open(output_file, 'a') as f:
                f.write(f"{video_path},{sign_id},{sign},{norm_x},{norm_y},{norm_w},{norm_h},{frame_number},{timestamp},{confidence}\n")

        # Optional: visualize
        for sign, color, x, y, w, h in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, sign, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Detection")
    parser.add_argument("--image", help="Process a single image")
    parser.add_argument("--inputfile", help="Process multiple images from a text file")
    parser.add_argument("--video", help="Process a video file")
    parser.add_argument("--output", default="output.txt", help="Output file name")

    args = parser.parse_args()

    if args.image:
        process_image(args.image, args.output)

    elif args.inputfile:
        with open(args.inputfile, 'r') as file:
            image_files = [line.strip() for line in file]
        for image_file in image_files:
            process_image(image_file, args.output)

    elif args.video:
        process_video(args.video, args.output)

    else:
        print("Please provide an input option: --image, --inputfile, or --video")
