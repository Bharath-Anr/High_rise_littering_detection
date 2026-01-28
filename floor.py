import cv2
import json
import os

# ---------------------------------------------------------
# Global state
# ---------------------------------------------------------
floor_boundaries = []
current_frame = None

# ---------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global floor_boundaries, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        floor_boundaries.append(y)
        print(f"Floor boundary marked at y = {y}")

        # Redraw all boundaries
        temp_frame = current_frame.copy()
        sorted_boundaries = sorted(floor_boundaries)

        for i, y_pos in enumerate(sorted_boundaries):
            cv2.line(
                temp_frame,
                (0, y_pos),
                (temp_frame.shape[1], y_pos),
                (0, 0, 255),
                2
            )
            cv2.putText(
                temp_frame,
                f"Floor {i + 1}",
                (10, y_pos - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Calibration", temp_frame)


# ---------------------------------------------------------
# Load first frame of video
# ---------------------------------------------------------
VIDEO_PATH = "videos/vid2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, current_frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read video frame for calibration")

# ---------------------------------------------------------
# OpenCV window setup
# ---------------------------------------------------------
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Calibration", mouse_callback)
cv2.imshow("Calibration", current_frame)

print("\nINSTRUCTIONS:")
print("- Click on each floor boundary from TOP to BOTTOM")
print("- Press 's' to save")
print("- Press 'q' to quit without saving\n")

# ---------------------------------------------------------
# Keyboard loop
# ---------------------------------------------------------
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit without saving.")
        break

    elif key == ord('s'):
        if len(floor_boundaries) == 0:
            print("No floor boundaries marked. Nothing to save.")
            break

        # Sort boundaries top → bottom
        floor_boundaries.sort()

        # Extract video ID (e.g. vid1 from vid1.mp4)
        video_id = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

        # Load existing calibration file if present
        if os.path.exists("floor_calibration.json"):
            with open("floor_calibration.json", "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Ensure top-level key exists
        if "boundaries" not in data:
            data["boundaries"] = {}

        # Store boundaries for this video
        data["boundaries"][video_id] = floor_boundaries

        # Save back to file
        with open("floor_calibration.json", "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"Saved {len(floor_boundaries)} floor boundaries "
            f"under boundaries['{video_id}']"
        )
        break

cv2.destroyAllWindows()
