"""
Export YOLOv8 to ONNX format for faster CPU inference
Run this once to create optimized ONNX model

Installation (if not already done):
    pip install onnx onnxruntime
"""

try:
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print("❌ Missing dependencies!")
    print("\nPlease install:")
    print("  pip install onnx onnxruntime")
    print("\nThen run this script again.")
    exit(1)

print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

print("Exporting to ONNX format...")
print("This will take 1-2 minutes...")

# Export with optimizations for CPU
model.export(
    format='onnx',
    imgsz=256,           # Match our input size (very small for speed)
    simplify=True,       # Simplify ONNX graph for better performance
    opset=12,            # ONNX opset version (compatible with most systems)
    dynamic=False        # Fixed input size (faster inference)
)

print("✓ Export complete!")
print("✓ Created: yolov8n.onnx")
print("\nNow use 'yolov8n.onnx' instead of 'yolov8n.pt' for 2-4x speedup on CPU")
