import cv2
import numpy as np
from collections import defaultdict

class FallingObjectDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.clahe = cv2.createCLAHE(    #CLAHE- CONTRAST ALTERNATIVE
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        # ---------------- FRAME DIFFERENCE ----------------
        self.prev_frame = None
        self.prev_prev_frame = None

        # ---------------- OPTICAL FLOW ----------------
        self.prev_gray = None
        self.prev_points = None

        # ---------------- TRACKING ----------------
        self.tracks = {}                 # track_id -> (cx, cy)
        self.trajectories = defaultdict(list)
        self.next_id = 0

        # ---------------- PARAMETERS ----------------
        self.MIN_AREA = 15
        self.MAX_AREA = 6000
        self.MIN_FALL_DISTANCE = 45
        self.MIN_VELOCITY = 2
        self.MAX_HORIZONTAL_DRIFT = 300
        self.MAX_MATCH_DIST = 60

        self.frame_count = 0

    # ==================================================
    # FRAME DIFFERENCING (MEDIUM + LARGE OBJECTS)
    # ==================================================
    def detect_motion_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if self.prev_frame is None or self.prev_prev_frame is None:
            if self.prev_frame is None:
                self.prev_frame = gray
            else:
                self.prev_prev_frame = self.prev_frame
                self.prev_frame = gray
            return None

        diff1 = cv2.absdiff(self.prev_prev_frame, self.prev_frame)
        diff2 = cv2.absdiff(self.prev_frame, gray)
        motion = cv2.bitwise_and(diff1, diff2)

        _, thresh = cv2.threshold(motion, 12, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=3)

        self.prev_prev_frame = self.prev_frame
        self.prev_frame = gray

        return thresh

    # ==================================================
    # OPTICAL FLOW (SMALL OBJECTS)
    # ==================================================
    def detect_optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray) #selective contrast adjustment

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=300, qualityLevel=0.01,
                minDistance=5, blockSize=7
            )
            return []

        if self.prev_points is None or len(self.prev_points) < 20:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=300, qualityLevel=0.01,
                minDistance=5, blockSize=7
            )
            self.prev_gray = gray
            return []

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        ) # type: ignore

        detections = []
        new_points = []

        for i, (new, old) in enumerate(zip(next_pts, self.prev_points)):
            if status[i] == 0:
                continue 

            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            dy = y_new - y_old
            dx = abs(x_new - x_old)

            if dy > 2 and dx < 20:
                detections.append((int(x_new), int(y_new)))
                new_points.append([[x_new, y_new]])

        self.prev_gray = gray
        self.prev_points = np.array(new_points, dtype=np.float32) if new_points else None

        return detections

    # ==================================================
    # FALL LOGIC
    # ==================================================
    def is_falling(self, trajectory):
        if len(trajectory) < 20:
            return False

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        net_fall = ys[-1] - ys[0]
        if net_fall < self.MIN_FALL_DISTANCE:
            return False

        downward = 0
        upward = 0
        total_down = 0

        for i in range(1, len(ys)):
            dy = ys[i] - ys[i - 1]
            if dy > 0:
                downward += 1
                total_down += dy
            elif dy < 0:
                upward += 1

        total_steps = downward + upward
        if total_steps == 0:
            return False

        # ---- RELAXED BUT SAFE RULES ----
        if downward / total_steps < 0.6:
            return False

        if upward > downward:
            return False

        velocity = net_fall / len(trajectory)
        if velocity < 1.2:
            return False

        if abs(xs[-1] - xs[0]) > self.MAX_HORIZONTAL_DRIFT:
            return False

        return True


    # ==================================================
    # FRAME PROCESSING
    # ==================================================
    def process_frame(self, frame):
        self.frame_count += 1
        display = frame.copy()

        detections = []

        # ---- LARGE OBJECTS ----
        mask = self.detect_motion_mask(frame)
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if self.MIN_AREA < area < self.MAX_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w // 2, y + h // 2
                    detections.append((cx, cy))
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # ---- SMALL OBJECTS ----
        detections.extend(self.detect_optical_flow(frame))

        # ---- TRACKING (CRITICAL FIX) ----
        new_tracks = {}
        used = set()

        for tid, (px, py) in self.tracks.items():
            best_i = None
            best_dist = float("inf")

            for i, (cx, cy) in enumerate(detections):
                if i in used:
                    continue
                if cy < py - 10:
                    continue

                dist = np.hypot(cx - px, cy - py)
                if dist < best_dist and dist < self.MAX_MATCH_DIST:
                    best_dist = dist
                    best_i = i

            if best_i is not None:
                cx, cy = detections[best_i]
                new_tracks[tid] = (cx, cy)
                self.trajectories[tid].append((cx, cy))
                used.add(best_i)

        for i, (cx, cy) in enumerate(detections):
            if i not in used:
                new_tracks[self.next_id] = (cx, cy)
                self.trajectories[self.next_id].append((cx, cy))
                self.next_id += 1

        self.tracks = new_tracks

        # ---- DRAW TRAJECTORIES ----
        fall_count = 0
        for tid, traj in self.trajectories.items():
            if len(traj) < 2:
                continue

            falling = self.is_falling(traj)
            color = (0, 0, 255) if falling else (150, 150, 150)
            thickness = 3 if falling else 1

            for i in range(1, len(traj)):
                cv2.line(display, traj[i - 1], traj[i], color, thickness)

            if falling:
                fall_count += 1
                x, y = traj[-1]
                cv2.putText(display, f"FALL #{tid}", (x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display,
                    f"Frame: {self.frame_count} | Falling objects: {fall_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        return display, mask

    # ==================================================
    # RUN
    # ==================================================
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            display, mask = self.process_frame(frame)

            cv2.imshow("Falling Object Detection", display)
            if mask is not None:
                cv2.imshow("Motion Mask", mask)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


# =========================
# USAGE
# =========================
if __name__ == "__main__":
    VIDEO_PATH = "videos/office2.mp4"   # change path
    detector = FallingObjectDetector(VIDEO_PATH)
    detector.run()
