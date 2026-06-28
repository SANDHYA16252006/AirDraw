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

# Bigger camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_x, prev_y = 0, 0
canvas = None

# Default brush settings
draw_color = (0, 255, 0)
brush_thickness = 8
eraser_thickness = 35
tool_name = "GREEN"

# Create full-screen window only once
cv2.namedWindow("AirDraw - Color Menu", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "AirDraw - Color Menu",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


def fingers_up(hand_landmarks):
    index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    return index_up, middle_up


def draw_menu(frame):
    height, width, _ = frame.shape

    # Menu background
    cv2.rectangle(frame, (0, 0), (width, 90), (40, 40, 40), -1)

    # BLUE button
    cv2.rectangle(frame, (10, 15), (130, 75), (255, 0, 0), -1)
    cv2.putText(
        frame, "BLUE", (28, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # GREEN button
    cv2.rectangle(frame, (145, 15), (285, 75), (0, 255, 0), -1)
    cv2.putText(
        frame, "GREEN", (155, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

    # RED button
    cv2.rectangle(frame, (300, 15), (420, 75), (0, 0, 255), -1)
    cv2.putText(
        frame, "RED", (330, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # ERASER button
    cv2.rectangle(frame, (435, 15), (590, 75), (220, 220, 220), -1)
    cv2.putText(
        frame, "ERASER", (445, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
    )

    # CLEAR button
    cv2.rectangle(frame, (605, 15), (760, 75), (80, 80, 80), -1)
    cv2.putText(
        frame, "CLEAR", (625, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )


while True:
    success, frame = cap.read()

    if not success:
        print("Camera could not be opened.")
        break

    frame = cv2.flip(frame, 1)

    # Get frame height and width
    height, width, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    draw_menu(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            index_up, middle_up = fingers_up(hand_landmarks)

            # SELECT MODE: Index + Middle finger
            if index_up and middle_up:

                # BLUE
                if 10 < x < 130 and 15 < y < 75:
                    draw_color = (255, 0, 0)
                    tool_name = "BLUE"

                # GREEN
                elif 145 < x < 285 and 15 < y < 75:
                    draw_color = (0, 255, 0)
                    tool_name = "GREEN"

                # RED
                elif 300 < x < 420 and 15 < y < 75:
                    draw_color = (0, 0, 255)
                    tool_name = "RED"

                # ERASER
                elif 435 < x < 590 and 15 < y < 75:
                    draw_color = (0, 0, 0)
                    tool_name = "ERASER"

                # CLEAR
                elif 605 < x < 760 and 15 < y < 75:
                    canvas = np.zeros_like(frame)
                    tool_name = "GREEN"

                prev_x, prev_y = 0, 0

            # DRAW MODE: Only index finger
            elif index_up and not middle_up:

                if prev_x != 0 and prev_y != 0:

                    if tool_name == "ERASER":
                        cv2.line(
                            canvas,
                            (prev_x, prev_y),
                            (x, y),
                            (0, 0, 0),
                            eraser_thickness
                        )
                    else:
                        cv2.line(
                            canvas,
                            (prev_x, prev_y),
                            (x, y),
                            draw_color,
                            brush_thickness
                        )

                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

    output = cv2.addWeighted(frame, 1, canvas, 1, 0)

    cv2.putText(
        output,
        "Tool: " + tool_name,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Index = Draw | Index + Middle = Select | Q = Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("AirDraw - Color Menu", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()