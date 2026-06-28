import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Previous finger position
prev_x, prev_y = 0, 0

# Drawing canvas
canvas = None

while True:
    success, frame = cap.read()

    if not success:
        print("Camera could not be opened.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Create canvas once
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks on hand
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get index finger tip landmark number 8
            index_tip = hand_landmarks.landmark[8]

            height, width, _ = frame.shape

            # Convert landmark position to screen pixels
            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            # Draw green line from previous point to current point
            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 8)

            # Update previous point
            prev_x, prev_y = x, y

    else:
        # Reset when hand is not visible
        prev_x, prev_y = 0, 0

    # Combine camera frame and drawing canvas
    output = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # Add text
    cv2.putText(
        output,
        "AirDraw - Move Index Finger to Draw | Press C to Clear | Q to Quit",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.imshow("AirDraw Step 3", output)

    key = cv2.waitKey(1) & 0xFF

    # Press C to clear drawing
    if key == ord("c"):
        canvas = np.zeros_like(frame)
        prev_x, prev_y = 0, 0

    # Press Q to quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()