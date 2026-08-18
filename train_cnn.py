from ultralytics import YOLO

# On Windows, multiprocessing requires this block to protect the script
if __name__ == '__main__':
    
    # 1. Load the Nano model
    model = YOLO('yolov8n.pt') 

    # 2. Train the model on your dataset
    results = model.train(
        # I grabbed your exact path from your error log!
        data='D:/VS Code/Python/CNN/data.yaml', 
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4 # Lowering workers slightly makes Windows happier
    )

    # 3. Export to an Edge format (Crucial for the Hackathon)
    success = model.export(format='onnx')
    print("Model exported to ONNX format successfully!")