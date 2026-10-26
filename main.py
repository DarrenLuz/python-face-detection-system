
import cv2
import os


def main():
    # Load the pre-trained frontal-face detector
    cascade_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Could not load the face detector.")
        return

    # Open the default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Check camera permissions or try camera index 1.")
        return

    print("Face detection started. Press Q to quit.")

    try:
        while True:
            # Read a frame from the webcam
            success, frame = camera.read()

            if not success:
                print("Error: Could not read a frame from webcam.")
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face locations
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Face detected",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Display the total number of faces
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # Display the webcam frame
            cv2.imshow("Face Detection System", frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Always release resources
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
