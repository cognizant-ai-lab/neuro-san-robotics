# VisionCore Performance Optimization Guide

## 🚀 Quick Start: Enable ONNX for 2-4x Speedup (MacBook)

```bash
# 1. Install ONNX dependencies
pip install onnx onnxruntime

# 2. Export YOLO to ONNX (one-time, ~2 minutes)
python export_yolo_onnx.py

# 3. Run VisionCore - it will auto-use ONNX
python coded_tools/unigo2/vision_core.py
```

**Expected improvement**: 5 FPS → 15-25 FPS on MacBook! ⚡

---

## Current Performance Optimizations (Applied)

### 1. **ONNX Runtime for CPU (MacBook) - CRITICAL** ⚡⚡⚡⚡
- **Problem**: PyTorch on CPU is very slow (5-8 FPS)
- **Solution**: Export YOLO to ONNX format (2-4x faster on CPU)
- **How to enable**: Run `python export_yolo_onnx.py` (one-time, 2 minutes)
- **Speed Gain**: 2-4x faster inference on MacBook/Desktop CPUs
- **After ONNX**: 15-25 FPS (vs 5-8 FPS with PyTorch)
- **Trade-off**: None! Same accuracy, just faster
- **Note**: Auto-detects and uses ONNX if yolov8n.onnx exists

### 2. **Reduced YOLO Input Size: 640 → 256**
- **Before**: 640x640 input to YOLO
- **After**: 256x256 input to YOLO (very aggressive)
- **Speed Gain**: ~6x faster (6x fewer pixels to process)
- **Trade-off**: May miss small/distant objects, lower accuracy

### 3. **Frame Downscaling: Capture Full, Process Small**
- **Before**: Processing full webcam resolution (1280x720 or higher)
- **After**: Capture at native resolution → Downscale to 320x240 → Process
- **Speed Gain**: 2-3x faster (fewer pixels to process)
- **Why better than reducing webcam res**: Avoids driver overhead
- **Bonus**: Bounding boxes scaled back to original frame size (accurate display)
- **Trade-off**: Detection on smaller frame, but display stays sharp

### 4. **Process Every 3rd Frame (with ONNX) or 5th (without)**
- **Before**: Processing every frame (30 FPS capture = 30 detections/sec)
- **After ONNX**: Processing every 3rd frame (~10 detections/sec)
- **After PyTorch**: Processing every 5th frame (~6 detections/sec)
- **Speed Gain**: Smoother video display, less compute load
- **Result**: Display runs at 30 FPS, detection at 6-10 FPS

### 3. **Webcam Resolution: Use Default**
- **Note**: Forcing lower webcam resolution can actually SLOW DOWN capture
- **Reason**: Driver overhead, non-native resolutions
- **Current**: Using webcam's default/native resolution
- **Alternative**: Downscale frames AFTER capture (see process_frame resize)

### 4. **Face Recognition Interval: Every 10 Detections**
- **Before**: Face recognition every frame (400ms each)
- **After**: Face recognition every 10th detection
- **Speed Gain**: 10x reduction in face processing
- **Press 'F' to disable completely**: Instant 2-3x speedup

---

## Expected FPS by Hardware & Backend

| Hardware | PyTorch (slow) | ONNX (fast) | TensorRT (fastest) | With Faces |
|----------|----------------|-------------|--------------------|------------|
| **MacBook Pro M1/M2** | 5-8 FPS | 15-25 FPS ⚡ | N/A | 10-18 FPS |
| **MacBook Pro Intel** | 3-6 FPS | 10-18 FPS ⚡ | N/A | 8-12 FPS |
| **Jetson Orin** | N/A | N/A | 60-100+ FPS ⚡ | 25-40 FPS |
| **Desktop RTX 3060** | 25-35 FPS | 35-50 FPS | 50-70 FPS ⚡ | 30-45 FPS |

**Key Takeaways:**
- ✅ MacBook: **Must use ONNX** for acceptable performance (2-4x faster)
- ✅ Jetson: **Must use TensorRT** for real-time performance (10-20x faster)
- ✅ Desktop GPU: TensorRT best, ONNX good fallback

---

## If Still Too Slow (< 10 FPS on MacBook)

### Option 1: Use Even Smaller Input Size
Edit `vision_core.py` line 945:
```python
input_size=256,  # Was 320, now even smaller
```
**Speed**: +30-50% faster, but detection quality suffers

### Option 2: Process Fewer Frames
Edit line 1033:
```python
if frame_count % 10 == 0:  # Was 5, now every 10th frame
```
**Speed**: 2x smoother video, but ~3 detections per second

### Option 3: Downscale Webcam Frames Before Processing
Edit line 1048-1049, uncomment:
```python
process_frame = cv2.resize(frame, (480, 360))  # Downscale to 75%
# process_frame = frame  # Comment this out
```
**Speed**: ~2x faster inference

### Option 4: Disable Face Recognition Permanently
In demo, set:
```python
face_recognition_interval = 0  # Line 1008, set to 0
```
Or just press `F` key during runtime.

---

## MacBook CPU Bottleneck Explanation

**Why is MacBook slow?**

1. **No GPU Acceleration**
   - YOLO runs on CPU (Intel/M1/M2)
   - PyTorch CPU inference is 10-20x slower than GPU
   - Jetson uses TensorRT + GPU = massive speedup

2. **YOLO is Compute-Heavy**
   - Even YOLOv8n (smallest) has ~3M parameters
   - Each inference: millions of multiply-add operations
   - MacBook CPU: ~50-100ms per frame at 320x320

3. **Face Recognition is Even Slower**
   - DeepFace loads large CNN models
   - 128D/512D embedding extraction
   - Database search for each face
   - MacBook CPU: 200-500ms per face

---

## Optimization Strategy

```
┌─────────────────────────────────────────────────────────┐
│ Video Pipeline: 30 FPS capture                         │
└────┬────────────────────────────────────────────────────┘
     │
     ├──> Frame 1  ─── DISPLAY (no processing)
     ├──> Frame 2  ─── DISPLAY (no processing)
     ├──> Frame 3  ─── DISPLAY (no processing)
     ├──> Frame 4  ─── DISPLAY (no processing)
     ├──> Frame 5  ─── PROCESS: YOLO (80ms)
     │                  └──> DISPLAY with boxes
     ├──> Frame 6  ─── DISPLAY (reuse Frame 5 boxes)
     ├──> Frame 7  ─── DISPLAY (reuse Frame 5 boxes)
     ├──> Frame 8  ─── DISPLAY (reuse Frame 5 boxes)
     ├──> Frame 9  ─── DISPLAY (reuse Frame 5 boxes)
     ├──> Frame 10 ─── PROCESS: YOLO (80ms)
     │                  └──> DISPLAY with boxes
     ├──> Frame 15 ─── PROCESS: YOLO (80ms) + DeepFace (400ms)
     │                  └──> DISPLAY with boxes + faces
     └──> ...

Result: 30 FPS smooth video, 6 detections/sec, 0.6 faces/sec
```

---

## Real-Time Performance on Jetson Orin

On the robot's Jetson Orin with TensorRT, you'll get:

```python
vision = VisionCore(
    yolo_model="yolov8n.pt",
    use_tensorrt=True,       # ⚡ Key optimization
    input_size=640,          # Can use full resolution
    half_precision=True,     # ⚡ FP16 acceleration
    confidence_threshold=0.65
)

# Process EVERY frame (no skipping needed)
if frame_count % 1 == 0:  # Every frame!
```

**Expected Performance:**
- YOLO: 5-10ms per frame (100-200 FPS capable)
- Face Recognition: 30-50ms (20-30 FPS)
- Combined: 40-60ms per frame = **25-30 FPS real-time**

---

## Quick Reference: Press Keys During Runtime

| Key | Action | Effect |
|-----|--------|--------|
| **F** | Toggle face recognition | 2-3x speed boost when OFF |
| **V** | Verbose mode | Show detection details |
| **S** | Save detection | Save current frame with boxes |
| **A** | Add face | Add person to database |
| **Q** | Quit | Exit program |

---

## 🔧 Bug Fixes Applied

### Bounding Box Alignment Issue (FIXED ✓)
**Problem**: Bounding boxes were not centered on detected objects
**Cause**: Processing downscaled frame (320x240) but drawing boxes on original frame (1280x720)
**Solution**: Scale bounding box coordinates back up to match original frame size
```python
# Automatically scales boxes from processed frame to display frame
scale_x = original_width / processed_width
scale_y = original_height / processed_height
bbox_display = [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
```
**Status**: Fixed in vision_core.py lines 1109-1124

### Webcam Resolution Slowdown (FIXED ✓)
**Problem**: Setting webcam to 640x480 made performance WORSE (3.5 FPS)
**Cause**: Driver overhead when forcing non-native resolutions
**Solution**: Capture at native resolution, downscale in software after capture
**Status**: Reverted to native capture, downscale in code

---

## Troubleshooting

### "Still only getting 5-8 FPS on MacBook"

Try these in order:

1. **Check Detection Time**
   - Look at bottom of screen: `Detection: XXXms`
   - Should be 50-80ms for YOLO only
   - If > 100ms, try smaller input_size (256)

2. **Check Face Recognition Status**
   - Press `F` to turn OFF
   - Should see immediate FPS increase (2-3x)
   - If no change, face recognition wasn't the bottleneck

3. **Reduce Frame Processing Interval**
   - Edit line 1033: `if frame_count % 10 == 0:`
   - Fewer detections per second, but smoother video

4. **Check System Load**
   - Close other apps (Chrome, Slack, etc.)
   - MacBook thermal throttling can slow down CPU
   - Activity Monitor: check CPU usage

### "Detection quality is poor with input_size=320"

This is expected. Small input size misses:
- Small objects (distant people, small items)
- Multiple objects in cluttered scenes
- Objects at weird angles

**Solutions:**
- Increase input_size to 416 or 480 (slower but better)
- Use yolov8s.pt instead of yolov8n.pt (more accurate, slightly slower)
- Accept trade-off: speed vs accuracy

---

## For Production on Go2 Robot

**Recommended Configuration:**

```python
# On Jetson Orin (coded_tools/unigo2/vision_core.py)
vision = VisionCore(
    yolo_model="yolov8n.pt",
    face_model="Facenet",
    use_tensorrt=True,           # ✓ Must enable
    confidence_threshold=0.65,    # ✓ Reduces false positives
    iou_threshold=0.45,
    input_size=640,              # ✓ Full quality
    half_precision=True          # ✓ FP16 for speed
)

# Process every frame or every 2nd frame
# Jetson can handle it!
```

**Expected Result:**
- Real-time object detection: 60-100 FPS
- With face recognition: 25-40 FPS
- Low latency: 10-40ms per frame
- Smooth robot operation

---

## Summary

| Optimization | Speed Gain | Quality Loss | Recommended For |
|--------------|------------|--------------|-----------------|
| Input size 320 | ⚡⚡⚡⚡ | ⚠️⚠️ Small objects | MacBook testing |
| Process every 5th frame | ⚡⚡⚡ | ⚠️ Lag | MacBook testing |
| Webcam 640x480 | ⚡⚡ | ⚠️ Lower res | MacBook testing |
| Face every 10 frames | ⚡⚡⚡ | ⚠️ Delayed ID | MacBook testing |
| TensorRT + Jetson | ⚡⚡⚡⚡⚡ | ✓ None | Production robot |

**Bottom Line for MacBook:**
- Development/testing only (not production)
- 10-15 FPS is acceptable for debugging
- Deploy to Jetson Orin for real-time performance

**Bottom Line for Jetson Orin:**
- Production-ready performance
- Real-time object detection + face recognition
- 25-40 FPS with full quality
