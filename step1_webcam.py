import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Camera could not be opened.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "AirDraw - Webcam Test",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AirDraw Webcam", frame)

    # Press Q to close camera
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()