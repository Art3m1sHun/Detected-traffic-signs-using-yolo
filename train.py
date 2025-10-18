from ultralytics import YOLO

# Put all the code that should only run once into this block
if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')

    # Train the model
    # The 'r' before the string path is good practice on Windows
    result = model.train(data=r'D:\Python\traffic_sign\signal.yaml', 
                         epochs=50, 
                         batch=8, 
                         imgsz = 320, 
                         device='cuda')