import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT.joinpath('runs/detect/train/weights/best.pt')


def scale_image(image, scale_percent):
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)
    new_dimensions = (new_width, new_height)
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image


def main():
    # Încarcă modelul YOLO
    model = YOLO(MODEL_PATH)

    # Deschide camera (0 = prima cameră găsită)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Nu s-a putut deschide camera.")
        return

    print("Camera pornită. Apasă 'q' pentru a închide.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠Eroare la citirea cadrelor de la cameră.")
            break

        # Rulează modelul YOLO pe cadrul actual
        results = model(frame, conf=0.55)

        # Desenează rezultatele pe imagine
        res_plotted = results[0].plot()

        # Afișează imaginea redimensionată
        cv2.imshow("Live", scale_image(res_plotted, 70))

        # Iese din buclă dacă se apasă 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Eliberare resurse
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
