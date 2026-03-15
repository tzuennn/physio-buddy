# Physio Buddy Debugging Guide

## Issue: Squat Count Not Increasing, Form/Fatigue Shows "-"

### Root Cause

MediaPipe is likely failing to detect your pose. This happens when:

1. **MediaPipe Not Installed**
   - The vision module requires: `pip install mediapipe opencv-python-headless`
   - Run: `pip install -e .[vision]`

2. **Poor Camera Setup**
   - Person not fully visible in frame
   - Camera distance too far (>3m) or too close (<0.5m)
   - Person facing away from camera (need side view for squats)
   - Poor lighting conditions

3. **Frame Quality Issues**
   - Blurry images
   - Motion blur from fast movements
   - Low JPEG quality encoding

### How to Debug

#### Step 1: Check Browser Console

Open Developer Tools (F12 or Right-click → Inspect) and go to Console tab.

You should see logs like:

```
Frame processed: {
  phase: "STAND",
  reps: 0,
  angles: { knee: 165.4, torso: 5.2, kneeOffset: -0.045 },
  form: { depth_quality: "good", ... },
  fatigue: "LOW"
}
```

If you see errors like:

- `"Frame error: No pose detected in frame"` → MediaPipe can't see your posture
- `"Frame error: Key landmarks not visible"` → Body parts blocked or out of frame

#### Step 2: Verify MediaPipe Installation

```bash
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import cv2; print('OpenCV OK')"
```

If these fail, install:

```bash
pip install -e .[vision]
pip install mediapipe opencv-python-headless
```

#### Step 3: Check Camera Setup

The app needs to see:

- **Full body**: Head to feet visible
- **Side view**: Standing sideways to camera (profile view)
- **Distance**: 1-2 meters away
- **Lighting**: Well-lit, no shadows on body
- **Background**: Clear background, avoid clutter

**Bad Setup:**

```
❌ Standing facing camera (front view)
❌ Only upper body visible
❌ Too close (0.3m away)
❌ Dark/shadowy lighting
❌ Back to camera
```

**Good Setup:**

```
✓ Standing sideways (left or right profile)
✓ Full body in frame
✓ 1-2 meters away
✓ Bright, even lighting
✓ Facing camera with left/right side visible
```

#### Step 4: Test with Browser Console

1. Open DevTools Console
2. Open the Network tab
3. Click "Start Session"
4. Do some squats
5. Watch the Console for logs

Each successful frame should show:

```
Frame processed: {
  phase: "BOTTOM",        // or STAND, DESCEND, ASCEND
  reps: 3,                // rep count increases
  angles: { knee: 95.2, torso: 12.3, ... }
  form: { ... }
  fatigue: "LOW"
}
```

#### Step 5: Check Network Requests

In DevTools Network tab:

1. Look for POST requests to `/sessions/{id}/ingest`
2. Click one, check "Response" tab
3. Should see frame metrics like:
   ```json
   {
     "frame": {
       "knee_angle_deg": 95.0,
       "torso_lean_deg": 15.0,
       "knee_inward_offset": -0.05
     },
     "phase": "BOTTOM",
     "rep_count": 5
   }
   ```

### Error Messages & Solutions

| Error Message                             | Cause                    | Solution                                                   |
| ----------------------------------------- | ------------------------ | ---------------------------------------------------------- |
| "Posture not detected"                    | MediaPipe can't see body | Move to better lighting, stand sideways                    |
| "Key landmarks not visible"               | Body parts out of frame  | Move closer, adjust camera angle, ensure full body visible |
| "Checking form... Try different lighting" | Poor image quality       | Increase brightness, remove shadows                        |
| "Connection error"                        | Network/API issue        | Check server is running, refresh page                      |
| "No active session"                       | Session expired/crashed  | Restart browser, try again                                 |

### Checking API Server

```bash
# Test health endpoint
curl http://localhost:8000/health
# Should return: {"status":"ok"}

# Test MediaPipe/Vision availability
python -c "
from physio_buddy.mediapipe_pose import MediaPipePoseAnalyzer
try:
    analyzer = MediaPipePoseAnalyzer()
    print('✓ MediaPipe analyzer initialized')
except Exception as e:
    print(f'✗ MediaPipe error: {e}')
"
```

### Angle Reference for Debugging

When standing:

- **Knee angle**: ~175-180° (straight leg)
- **Torso lean**: ~5-15° (upright)
- **Knee offset**: ~-0.2 to 0.1 (feet under body)

When at bottom of squat:

- **Knee angle**: ~90-110° (bent knee)
- **Torso lean**: ~20-30° (slight lean)
- **Knee offset**: Similar or more negative (knee might track inward)

### State Machine Thresholds

The squat phase tracker uses:

- **Descend threshold**: < 150° (start going down)
- **Bottom threshold**: ≤ 110° (at bottom of squat)
- **Ascend threshold**: > 135° (coming back up)
- **Stand threshold**: ≥ 150° (fully standing)

If reps don't count, angles might not be crossing these thresholds.

### Common Issues & Fixes

**Issue: "Squat count stays at 0"**

- Check that knee angle goes below 135° during squat
- Verify camera can see knee bending
- Ensure full depth squats (aim for 90-100° knee angle)

**Issue: "Form/Fatigue always shows -"**

- Same as above - pose detection failing
- Check console errors
- Verify MediaPipe installed

**Issue: "Form warnings but no reps counted"**

- Squat might be too shallow (not reaching 110° knee angle)
- Or too fast (frames skipped)
- Try slower, deeper squats

**Issue: "Stops after a few frames"**

- Camera permission might be revoked
- Brightness/lighting changed
- Check browser console for errors

### Advanced Debugging

Enable verbose logging:

```javascript
// In browser console:
localStorage.debug = "*";
location.reload();
```

Check frame timestamps in console:

```javascript
// In console, every frame should log with timestamp
console.log(new Date().toISOString());
```

### Still Not Working?

1. **Reinstall MediaPipe:**

   ```bash
   pip uninstall mediapipe opencv-python-headless
   pip install -e .[vision]
   ```

2. **Check Python version:**

   ```bash
   python --version  # Should be 3.9+
   ```

3. **Test MediaPipe directly:**

   ```bash
   python -c "
   import mediapipe as mp
   import cv2
   import numpy as np

   # Try to initialize
   pose = mp.solutions.pose.Pose(static_image_mode=True)

   # Generate dummy image
   dummy = np.zeros((480, 640, 3), dtype=np.uint8)
   result = pose.process(dummy)

   print(f'Landmarks: {result.pose_landmarks}')
   print('✓ MediaPipe working')
   "
   ```

4. **Check API logs:**
   - Look at FastAPI server terminal
   - Should show each request, any errors will appear there

5. **Try the endpoints directly:**

   ```bash
   # Start a session
   curl -X POST http://localhost:8000/sessions/start

   # Should get back: {"session_id":"...", "safety_notice":"..."}
   ```

### Performance Tips

- Reduce JPEG quality if server CPU high
- Increase frame send interval (currently 500ms)
- Reduce video resolution (currently 640x480)

To modify in app.js:

```javascript
const sendIntervalMs = 500; // Increase to 1000 for slower processing
```

---

**Last Resort:** Clear camera cache and try different browser

```bash
# Chrome: Clear site data for localhost:8000
# Firefox: Clear cache for site
```
