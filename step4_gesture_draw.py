import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
canvas = None

def fingers_up(hand_landmarks):
    """
    Returns True/False for:
    index finger and middle finger.
    """
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

    return index_up, middle_up


while True:
    success, frame = cap.read()

    if not success:
        print("Camera could not be opened.")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    mode_text = "Show your hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            height, width, _ = frame.shape

            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            index_up, middle_up = fingers_up(hand_landmarks)

            # DRAW MODE: only index finger is up
            if index_up and not middle_up:
                mode_text = "DRAW MODE"

                if prev_x != 0 and prev_y != 0:
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        (0, 255, 0),
                        8
                    )

                prev_x, prev_y = x, y

            # SELECT MODE: index + middle finger are up
            elif index_up and middle_up:
                mode_text = "SELECT MODE - Drawing Paused"
                prev_x, prev_y = 0, 0

            # HAND CLOSED / OTHER GESTURE
            else:
                mode_text = "Drawing Paused"
                prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

    output = cv2.addWeighted(frame, 1, canvas, 1, 0)

    cv2.putText(
        output,
        mode_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "Index only = Draw | Index + Middle = Pause | C = Clear | Q = Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("AirDraw Step 4", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        canvas = np.zeros_like(frame)
        prev_x, prev_y = 0, 0

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()