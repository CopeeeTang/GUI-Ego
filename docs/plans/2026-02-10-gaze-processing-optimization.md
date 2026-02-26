# Gaze Data Processing Optimization - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the preview system's eye tracking data processing by fixing bugs, passing `fixation id` through the full stack, and implementing fixation-based UI trigger timing with FOV safety constraints (inspired by StreamGaze).

**Architecture:** Three-layer fix: (1) Backend `server.py` passes `fixation_id` to frontend, (2) Frontend `video_overlay.html` uses fixation data to determine UI trigger timing based on annotation end time, (3) FOV safety region clamps UI position away from video edges. All changes are backward-compatible — samples without fixation data fall back to current behavior.

**Tech Stack:** Python (server.py), Vanilla JavaScript (video_overlay.html), CSV/JSON data files

---

### Task 1: Fix server.py to pass fixation_id in /api/gaze

**Files:**
- Modify: `agent/preview/server.py:2825-2853`

**Step 1: Add fixation_id to gaze JSON response**

In `serve_gaze_data()`, the current code only extracts `time_s`, `gaze x [px]`, `gaze y [px]`. Add `fixation id` and `worn`:

```python
# In serve_gaze_data(), replace the append block (lines 2845-2849):
gaze_data.append({
    'time': time_s,
    'x': float(x_str) if x_str else None,
    'y': float(y_str) if y_str else None,
    'fixation_id': float(row.get('fixation id', '').strip() or 'nan') if row.get('fixation id', '').strip() else None,
    'worn': row.get('worn', '').strip().lower() == 'true',
})
```

**Step 2: Verify the change**

Run: `cd /home/v-tangxin/GUI && source ml_env/bin/activate && python3 -c "
import json, csv
from pathlib import Path
p = Path('agent/example/Task2.2/P10_Ernesto/sample_001/signals/gaze.csv')
with open(p) as f:
    reader = csv.DictReader(f)
    row = next(reader)
    print('Columns:', list(row.keys()))
    print('fixation id value:', repr(row.get('fixation id', '')))
"`

Expected: Shows columns including `fixation id` with a numeric value like `6110.0`

**Step 3: Commit**

```bash
git add agent/preview/server.py
git commit -m "fix: pass fixation_id and worn fields through /api/gaze endpoint"
```

---

### Task 2: Fix frontend getGazeAt() binary search bug

**Files:**
- Modify: `agent/preview/templates/video_overlay.html:347-375`

**Step 1: Replace getGazeAt() with correct closest-point + interpolate logic**

Replace the entire `getGazeAt()` function (lines 347-375):

```javascript
function getGazeAt(time) {
    if (gazeData.length === 0) return null;

    // Binary search for closest point
    let low = 0, high = gazeData.length - 1;
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (gazeData[mid].time < time) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    // Choose the closer of adjacent points (fix: was missing this)
    if (low > 0 && Math.abs(gazeData[low].time - time) > Math.abs(gazeData[low - 1].time - time)) {
        low = low - 1;
    }

    const point = gazeData[low];
    if (point && point.x !== null && point.y !== null) {
        return { x: point.x, y: point.y, fixation_id: point.fixation_id };
    }

    // Fallback strategies
    const fallback = document.getElementById('gaze-fallback').value;

    if (fallback === 'last_valid') {
        for (let i = low - 1; i >= 0; i--) {
            if (gazeData[i].x !== null && gazeData[i].y !== null) {
                return { x: gazeData[i].x, y: gazeData[i].y, fixation_id: gazeData[i].fixation_id };
            }
        }
    } else if (fallback === 'interpolate') {
        // Find nearest valid points before and after
        let beforeIdx = -1, afterIdx = -1;
        for (let i = low - 1; i >= 0; i--) {
            if (gazeData[i].x !== null && gazeData[i].y !== null) { beforeIdx = i; break; }
        }
        for (let i = low; i < gazeData.length; i++) {
            if (gazeData[i].x !== null && gazeData[i].y !== null) { afterIdx = i; break; }
        }
        if (beforeIdx >= 0 && afterIdx >= 0) {
            const tBefore = gazeData[beforeIdx].time;
            const tAfter = gazeData[afterIdx].time;
            const tRange = tAfter - tBefore;
            if (tRange > 0) {
                const alpha = (time - tBefore) / tRange;
                return {
                    x: (1 - alpha) * gazeData[beforeIdx].x + alpha * gazeData[afterIdx].x,
                    y: (1 - alpha) * gazeData[beforeIdx].y + alpha * gazeData[afterIdx].y,
                    fixation_id: null
                };
            }
        }
        // Fall through to last_valid if interpolation fails
        for (let i = low - 1; i >= 0; i--) {
            if (gazeData[i].x !== null && gazeData[i].y !== null) {
                return { x: gazeData[i].x, y: gazeData[i].y, fixation_id: gazeData[i].fixation_id };
            }
        }
    }

    // Ultimate fallback: video center
    return { x: video.videoWidth / 2, y: video.videoHeight / 2, fixation_id: null };
}
```

**Step 2: Test in browser**

Run: `cd /home/v-tangxin/GUI && ./ml_env/bin/python3 -m agent.preview.server --port 8000`

Open a sample in the browser, verify gaze indicator moves smoothly and tracks correctly.

**Step 3: Commit**

```bash
git add agent/preview/templates/video_overlay.html
git commit -m "fix: correct binary search to find closest gaze point and implement interpolate fallback"
```

---

### Task 3: Implement fixation-based UI trigger timing

**Files:**
- Modify: `agent/preview/templates/video_overlay.html:174-344`

This is the core feature change. Replace the hardcoded "last N seconds" UI timing with fixation-based timing.

**Step 1: Add fixation detection helper function**

Add after the `getGazeAt()` function:

```javascript
// Find the first stable fixation after a given time
// Returns {startTime, endTime, centerX, centerY} or null
function findFirstFixationAfter(afterTime) {
    if (gazeData.length === 0) return null;

    let currentFixId = null;
    let fixPoints = [];

    for (let i = 0; i < gazeData.length; i++) {
        const pt = gazeData[i];
        if (pt.time < afterTime) continue;
        if (pt.x === null || pt.y === null) continue;

        if (pt.fixation_id !== null && pt.fixation_id !== undefined) {
            // Using Pupil Labs fixation id
            if (pt.fixation_id !== currentFixId) {
                // New fixation started
                if (fixPoints.length >= 3) {
                    // Previous fixation was valid (≥3 points ≈ 15ms at 200Hz)
                    return computeFixation(fixPoints);
                }
                currentFixId = pt.fixation_id;
                fixPoints = [pt];
            } else {
                fixPoints.push(pt);
            }
        }
    }

    // Check last fixation
    if (fixPoints.length >= 3) {
        return computeFixation(fixPoints);
    }

    return null;
}

function computeFixation(points) {
    const sumX = points.reduce((s, p) => s + p.x, 0);
    const sumY = points.reduce((s, p) => s + p.y, 0);
    return {
        startTime: points[0].time,
        endTime: points[points.length - 1].time,
        centerX: sumX / points.length,
        centerY: sumY / points.length,
        pointCount: points.length
    };
}
```

**Step 2: Add FOV safety clamping function**

```javascript
// Clamp UI position to FOV safe region (inspired by StreamGaze perifoveal bound)
// r_fov ≈ W * tan(8°) / tan(HFOV/2), for Pupil Labs Neon HFOV≈100°
function clampToSafeRegion(x, y, videoW, videoH) {
    const FOV_ANGLE_DEG = 8;
    const HFOV_DEG = 100; // Pupil Labs Neon horizontal FOV
    const rFov = videoW * Math.tan(FOV_ANGLE_DEG * Math.PI / 180) / Math.tan((HFOV_DEG / 2) * Math.PI / 180);
    const margin = rFov;

    return {
        x: Math.max(margin, Math.min(videoW - margin, x)),
        y: Math.max(margin, Math.min(videoH - margin, y))
    };
}
```

**Step 3: Modify updateUI() to use fixation-based timing**

Replace the `updateUI()` function (lines 321-344):

```javascript
// State for fixation-based UI trigger
let uiTriggerFixation = null;  // Cached fixation result
let uiTriggerComputed = false; // Whether we've computed it

function computeUITrigger() {
    if (uiTriggerComputed) return;
    uiTriggerComputed = true;

    // Get annotation end time from rawdata
    let annotationEndTime = null;
    if (rawData && rawData.time_interval) {
        annotationEndTime = rawData.time_interval.end || rawData.time_interval.start;
    }

    if (annotationEndTime !== null && gazeData.some(g => g.fixation_id !== null)) {
        // Convert absolute annotation time to relative gaze time
        // gaze.csv time_s is relative to recording start
        // rawdata.json time_interval is also relative to recording start
        uiTriggerFixation = findFirstFixationAfter(annotationEndTime);
        if (uiTriggerFixation) {
            console.log('UI trigger fixation found:', uiTriggerFixation);
        } else {
            console.warn('No fixation found after annotation end time:', annotationEndTime);
        }
    }
}

function updateUI() {
    const currentTime = video.currentTime;
    const duration = video.duration || 1;
    const uiDuration = parseFloat(document.getElementById('ui-duration').value);

    const progress = (currentTime / duration) * 100;
    timelineProgress.style.width = `${progress}%`;
    document.getElementById('current-time').textContent = formatTime(currentTime);
    document.getElementById('total-time').textContent = formatTime(duration);

    // Compute fixation trigger (once, after data is loaded)
    computeUITrigger();

    // Determine UI visibility window
    let uiStartTime, uiEndTime;

    if (uiTriggerFixation) {
        // Fixation-based: show from fixation start, for uiDuration seconds
        // Convert gaze time_s (absolute from recording) to video-relative time
        // Video covers rawData.time_interval.start to rawData.time_interval.end
        const videoOffset = (rawData && rawData.time_interval) ? rawData.time_interval.start : 0;
        const gazeTimeOffset = gazeData.length > 0 ? gazeData[0].time : 0;
        // Map: gazeTime -> videoTime = gazeTime - gazeTimeOffset (since video starts at gaze start)
        // But gaze CSV starts at annotation_start - 5s window
        // Video start aligns with gaze data start (both served from same time reference)
        uiStartTime = uiTriggerFixation.startTime - gazeData[0].time;
        uiEndTime = uiStartTime + uiDuration;

        document.getElementById('ui-window-info').textContent =
            `Fixation @${uiStartTime.toFixed(1)}s (+${uiDuration.toFixed(1)}s)`;
    } else {
        // Fallback: last N seconds (original behavior)
        uiStartTime = duration - uiDuration;
        uiEndTime = duration;
        document.getElementById('ui-window-info').textContent = `Last ${uiDuration.toFixed(1)}s`;
    }

    // Update timeline window indicator
    const windowStart = Math.max(0, (uiStartTime / duration) * 100);
    const windowWidth = Math.min(100 - windowStart, (uiDuration / duration) * 100);
    timelineUIWindow.style.left = `${windowStart}%`;
    timelineUIWindow.style.width = `${windowWidth}%`;

    // Show/hide UI
    if (currentTime >= uiStartTime && currentTime <= uiEndTime) {
        uiOverlay.classList.remove('hidden');
    } else {
        uiOverlay.classList.add('hidden');
    }

    updateGazePosition(currentTime);
}
```

**Step 4: Update updateGazePosition() with FOV clamping**

Replace the `updateGazePosition()` function (lines 378-407):

```javascript
function updateGazePosition(currentTime) {
    const videoEl = document.getElementById('video');
    const container = document.getElementById('video-container');
    const containerRect = container.getBoundingClientRect();
    const gaze = getGazeAt(currentTime + (gazeData.length > 0 ? gazeData[0].time : 0));
    if (!gaze) return;

    // Calculate actual video render area (handle letterboxing)
    const videoAspect = videoEl.videoWidth / videoEl.videoHeight;
    const containerAspect = containerRect.width / containerRect.height;
    let renderWidth, renderHeight, offsetX, offsetY;

    if (containerAspect > videoAspect) {
        // Container wider than video: letterbox on sides
        renderHeight = containerRect.height;
        renderWidth = renderHeight * videoAspect;
        offsetX = (containerRect.width - renderWidth) / 2;
        offsetY = 0;
    } else {
        // Container taller than video: letterbox on top/bottom
        renderWidth = containerRect.width;
        renderHeight = renderWidth / videoAspect;
        offsetX = 0;
        offsetY = (containerRect.height - renderHeight) / 2;
    }

    const scaleX = renderWidth / videoEl.videoWidth;
    const scaleY = renderHeight / videoEl.videoHeight;
    const displayX = offsetX + gaze.x * scaleX;
    const displayY = offsetY + gaze.y * scaleY;

    // Gaze indicator
    if (showGaze) {
        gazeIndicator.style.display = 'block';
        gazeIndicator.style.left = `${displayX}px`;
        gazeIndicator.style.top = `${displayY}px`;
    } else {
        gazeIndicator.style.display = 'none';
    }

    // UI position with FOV safety clamping
    const mode = document.getElementById('position-mode').value;
    if (mode === 'gaze') {
        // Clamp to safe region (avoid edges)
        const clamped = clampToSafeRegion(gaze.x, gaze.y, videoEl.videoWidth, videoEl.videoHeight);
        const clampedDisplayX = offsetX + clamped.x * scaleX;
        const clampedDisplayY = offsetY + clamped.y * scaleY;
        uiOverlay.style.left = `${clampedDisplayX}px`;
        uiOverlay.style.top = `${clampedDisplayY}px`;
    } else if (mode === 'center') {
        uiOverlay.style.left = `${containerRect.width / 2}px`;
        uiOverlay.style.top = `${containerRect.height / 2}px`;
    } else {
        uiOverlay.style.left = `${containerRect.width / 2}px`;
        uiOverlay.style.top = `${containerRect.height * 0.7}px`;
    }
}
```

**Step 5: Reset trigger cache on data reload**

In the `loadData()` function, after loading gaze data, add:

```javascript
// Reset fixation trigger cache when new data loads
uiTriggerComputed = false;
uiTriggerFixation = null;
```

And after `rawData = await rawResp.json();`, add:

```javascript
// Recompute UI trigger when rawdata loads
uiTriggerComputed = false;
```

**Step 6: Test with the preview server**

Run: `cd /home/v-tangxin/GUI && ./ml_env/bin/python3 -m agent.preview.server --port 8000`

Verify:
1. Open a sample with gaze data
2. Timeline should show the UI window starting at the fixation point (not at the end)
3. UI overlay should appear at the fixation moment during playback
4. UI position should be clamped away from edges
5. Samples without fixation data should fall back to "last N seconds" behavior

**Step 7: Commit**

```bash
git add agent/preview/templates/video_overlay.html
git commit -m "feat: implement fixation-based UI trigger timing with FOV safety clamping

Replaces hardcoded 'last N seconds' UI timing with intelligent fixation detection.
UI now triggers at the first stable fixation after the annotation end time.
FOV safety region (8° perifoveal bound) prevents UI from appearing at video edges.
Falls back to original behavior when fixation data is unavailable."
```

---

### Task 4: Fix Canvas recording coordinate mapping

**Files:**
- Modify: `agent/preview/templates/video_overlay.html:458-474`

**Step 1: Fix renderCompositeFrame() to use video coordinates**

Replace the `renderCompositeFrame()` function:

```javascript
async function renderCompositeFrame() {
    if (!compositeCtx) return;
    compositeCtx.drawImage(video, 0, 0, compositeCanvas.width, compositeCanvas.height);

    const overlay = document.getElementById('ui-overlay');
    if (overlay && !overlay.classList.contains('hidden') && typeof html2canvas !== 'undefined') {
        try {
            const uiCanvas = await html2canvas(overlay, { backgroundColor: null, scale: 1, logging: false, useCORS: true });

            // Use gaze coordinates directly in video space (not CSS coordinates)
            const currentTime = video.currentTime;
            const gaze = getGazeAt(currentTime + (gazeData.length > 0 ? gazeData[0].time : 0));
            let drawX, drawY;

            const mode = document.getElementById('position-mode').value;
            if (gaze && mode === 'gaze') {
                const clamped = clampToSafeRegion(gaze.x, gaze.y, compositeCanvas.width, compositeCanvas.height);
                drawX = clamped.x;
                drawY = clamped.y;
            } else {
                drawX = compositeCanvas.width / 2;
                drawY = mode === 'fixed' ? compositeCanvas.height * 0.7 : compositeCanvas.height / 2;
            }

            compositeCtx.drawImage(uiCanvas, drawX - uiCanvas.width / 2, drawY - uiCanvas.height / 2);
        } catch (e) {
            console.warn('html2canvas render failed:', e);
        }
    }
}
```

**Step 2: Commit**

```bash
git add agent/preview/templates/video_overlay.html
git commit -m "fix: use video coordinates for canvas recording overlay position"
```

---

### Task 5: End-to-end verification and time alignment validation

**Step 1: Verify data flow with a real sample**

Run:
```bash
cd /home/v-tangxin/GUI && source ml_env/bin/activate
python3 -c "
import json, csv
from pathlib import Path

# Load rawdata
raw = json.load(open('agent/example/Task2.2/P10_Ernesto/sample_001/rawdata.json'))
print('Annotation end time:', raw['time_interval']['end'])

# Load gaze and find first fixation after annotation end
end_time = raw['time_interval']['end']
with open('agent/example/Task2.2/P10_Ernesto/sample_001/signals/gaze.csv') as f:
    reader = csv.DictReader(f)
    current_fix = None
    fix_count = 0
    for row in reader:
        t = float(row['time_s'])
        fix_id = row.get('fixation id', '').strip()
        if t < end_time:
            continue
        if fix_id and fix_id != current_fix:
            if fix_count >= 3:
                print(f'First fixation after annotation: fix_id={current_fix}, at t={t:.3f}s, count={fix_count}')
                break
            current_fix = fix_id
            fix_count = 1
        elif fix_id == current_fix:
            fix_count += 1
    else:
        print('No fixation found after annotation end time')
"
```

Expected: Shows a fixation ID and timestamp after the annotation end time.

**Step 2: Visual verification with preview server**

Run: `./ml_env/bin/python3 -m agent.preview.server --port 8000`

Check:
- [ ] Gaze indicator tracks correctly (no jumps or drift)
- [ ] UI appears at the fixation point after annotation end time
- [ ] UI stays within safe FOV region (not at edges)
- [ ] Timeline shows correct UI window position
- [ ] Export/screenshot positions match preview
- [ ] Fallback works when fixation data is absent

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: add gaze processing research report and implementation plan"
```
