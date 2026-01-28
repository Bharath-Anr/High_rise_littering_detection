# High_rise_littering_detection
The falling object detector combines three complementary techniques:
  1.Frame Differencing → Detects medium & large moving objects
  2.Optical Flow (Lucas–Kanade) → Detects small fast-moving objects
  3.Trajectory-based Fall Logic → Confirms true falling motion over time
  
Falling Object Detection Pipeline
1. Frame Differencing (Medium & Large Objects)
Uses three-frame differencing
CLAHE + Gaussian blur for lighting robustness
Thresholding and dilation to extract motion regions
Purpose: Detect larger moving objects efficiently.

2. Optical Flow (Small Objects)
Uses Lucas–Kanade pyramidal optical flow
Tracks feature points frame-to-frame
Filters motion based on:
Downward displacement
Limited horizontal drift
Purpose: Detect small or fast-falling objects missed by frame differencing.

3. Object Tracking
* Matches detections using nearest-neighbor distance
* Maintains object IDs and trajectories
* Prevents backward jumps and duplicate tracks

4. Fall Detection Logic
A trajectory is considered a fall only if:
* Sufficient number of frames exist
* Net downward displacement exceeds a threshold
* Majority of motion is downward
* Average vertical velocity is high enough
* Horizontal drift is limited
This avoids false positives from slow or oscillating motion.


## Floor Boundary Calibration Tool
Purpose
Allows manual marking of floor levels for multi-floor or height-aware analysis.
How it works
1. Loads the first video frame
2. User clicks horizontal floor boundaries
3. Boundaries are labeled and displayed
4. Press:
    * s to save
    * q to quit without saving
Output (floor_calibration.json)
