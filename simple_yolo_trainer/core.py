import torch

from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).parent.parent
DATASETS = ROOT.joinpath('datasets/yolo')

MODEL_YOLOV7N = 'yolov7n'
MODEL_YOLOV8N = 'yolov8n'
MODEL_YOLOV8S = 'yolov8s'
MODEL_YOLOV10M = 'yolov10m'
MODEL_YOLOV11N = 'yolo11n'
MODEL_YOLOV11M = 'yolo11m'
MODEL_YOLOV8X = 'yolov8x'
MODEL_YOLOV8X_SEG = 'yolov8n-seg'
TRAINING_IMGSZ = 640
TRAINING_EPOCHS = 500

def train_model():
    """
    train_model
    :return:
    """

    dataset_directory = DATASETS.joinpath('example')
    data_yaml = dataset_directory.joinpath('data.yaml')

    # Load a pre-trained YOLO model (YOLOv8n is lightweight; you can use 'yolov8s.pt_old' for larger models)
    model = YOLO(MODEL_YOLOV8N)

    # Train the model
    # Change device according to your needs. Device = 0 -> it should use your GPU. to use CPU, set device as 'cpu'
    model.train(data=data_yaml, epochs=TRAINING_EPOCHS, imgsz=TRAINING_IMGSZ, device= 0, task='detect')
    #print('model device: ', model.device)

    # Validate the model
    results = model.val()

    # Save the best model
    model.export()  # Export for deployment if needed


def self_test():
    """
    self_test
    :return:
    """
    cuda_version = torch.version.cuda  # Shows the CUDA version PyTorch is using
    print(f'{cuda_version=}')
    cuda_available = torch.cuda.is_available()  # Should return True
    print(f'{cuda_available=}')
    device = torch.cuda.get_device_name(0)  # Should print the name of your GPU
    print(f'{device=}')


if __name__ == '__main__':
    """
    pass
    """
    self_test()
    train_model()
