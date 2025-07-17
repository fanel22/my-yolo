import cv2

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).parent.parent
DATASETS = ROOT.joinpath('datasets/yolo')


def scale_image(image, scale_percent):
    """
    scale_image
    """

    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image


def display_scaled(image, scale_percent, view='Image', blocking=False):
    """

    """
    cv2.imshow(view, scale_image(image, scale_percent))
    # cv2.VideoCapture(0, scale_image(image, scale_percent))
    if blocking:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # replace with your model
    model = YOLO(ROOT.joinpath('runs/detect/train/weights/best.pt'))

    test_images = [
        # manually add images paths here
    ]

    # or add entire folder of images (assumed images are .png)
    test_images_path = DATASETS.joinpath('example/train/images')

    test_images_dataset = [
        Path(img) for img in
        list(test_images_path.glob('*.jpg'))
    ]
    test_images.extend(test_images_dataset)
    # print(test_images)

    # test_images.extend(kaggle_dataset)
    for img in test_images[:1000]:
        results = model(img, conf=0.55)

        res_plotted = results[0].plot()
        for result in results:
            boxes = result.boxes
            masks = result.masks
            probs = result.probs
            print(boxes, masks, probs)

        display_scaled(res_plotted, 50, 'Test', blocking=True)
    cv2.destroyAllWindows()
