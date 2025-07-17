# simple_yolo_trainer
A lightweight, beginner-friendly toolkit for training YOLO object detection/segmentation models with minimal setup and clear instructions. Ideal for custom datasets and rapid experimentation.

⚙️ Prerequisites
Before you begin, make sure your system meets the following requirements:

Operating System: Windows (tested on Windows 11)

Python: 3.8 or higher (tested on 3.9.13)

NVIDIA GPU (Recommended):
  - Required for training with CUDA acceleration
  - Ensure NVIDIA drivers, CUDA Toolkit, and cuDNN are properly installed

pip: Python package installer

Git: For cloning the repository

💡 Note: Training without a GPU is possible but significantly slower and not recommended for large datasets or models.


## 📦 Installation
All commands can be ran in cmdline.
Follow the steps below to set up your environment and get started:

### 1. Clone the repository

```bash
git clone https://github.com/scorpionipx/simple_yolo_trainer
cd simple_yolo_trainer
```


### 2. Create a virtual environment

```bash
python -m venv .venv
```
OR (you might need to install virtualenv first into your Python distribution)
```bash
python -m virtualenv .venv
```

### 3. Activate the environment

```cmd
.venv\Scripts\activate
```

### 4. Install required packages
Once the virtual environment is activated, install all necessary Python dependencies using:

```cmd
pip install -r requirements.txt
```
This will ensure all required libraries (e.g., PyTorch, OpenCV, Ultralytics YOLO, etc.) are properly installed for training and evaluation.

✅ Tip: If you encounter any CUDA-related issues, verify your torch and torchvision versions are compatible with your installed CUDA version. You can check available options at https://pytorch.org/get-started/locally


### 5. Verify CUDA setup (optional but recommended)
To ensure your environment and GPU setup are working correctly, run the built-in self-test:

```bash
python simple_yolo_trainer/core.py
```

This will output:
- Your CUDA version
- Whether CUDA is available
- The name of the GPU being used

💡 Note: This does not start training — it just checks your PyTorch+CUDA setup (by default train_model method is commented out).



### 6. Add your dataset into project
![image](https://github.com/user-attachments/assets/24153c5b-dc8a-4ece-bcb6-ee20d332606a)
Split your dataset into 2 folders: train (80%) and val(20%)
✅ Tip: your dataset should contain the training images and their annotations files.

### 7. Create (if not already existing) data.yaml file (example provided)
- specify training and validation directories (absolute or relative paths)
- specify number of classes
- specify classes list


### 8. Train your model
Uncomment train_model() method in main.
Adjust parameters if needed.
Set your dataset_directory.
Change YOLO model, if required.

```bash
python simple_yolo_trainer/core.py
```

### 9. Test you model
After training your model, a **runs** folder will be created. 
Find your **weights/best.pt** model and load it into test_model.py.
Specify your test data (images).
Run test_model.py.


```bash
python simple_yolo_trainer/test_model.py
```



Every step can be followed with the already provided dataset and yaml examples.
