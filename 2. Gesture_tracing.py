#Programmer: Saad Syed
#Code description: code for the tracing exercise
#Last update date: 3/2/25
#Update description: objects to be traced updated

import cv2
import mediapipe as mp
import numpy as np
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()

def generate_random_shape(width, height):
    """Generate a random shape (circle, square, triangle) at a random position with a specific color."""
    shape_types = ["circle", "square", "triangle"]
    shape_type = random.choice(shape_types)
    x = random.randint(150, width - 150)
    y = random.randint(150, height - 150)
    shape_color = random.choice([(0, 0, 255), (0, 255, 0), (255, 0, 0)])  # Red, Green, Blue
    return shape_type, (x, y), shape_color

def draw_shape(frame, shape):
    """Draw the shape on the frame."""
    shape_type, position, color = shape
    x, y = position

    if shape_type == "circle":
        cv2.circle(frame, position, 80, color, 2)  
    elif shape_type == "square":
        cv2.rectangle(frame, (x - 80, y - 80), (x + 80, y + 80), color, 2)
    elif shape_type == "triangle":
        pts = np.array([[x, y - 80], [x - 80, y + 80], [x + 80, y + 80]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

def is_tracing_correct(traced_points, shape, current_color):
    """Check if the shape is **fully** traced with the correct color."""
    shape_type, position, shape_color = shape
    if current_color != shape_color:
        return False  # Ensure correct color is used

    return len(traced_points) > 500  # Require full shape tracing

def create_color_buttons(frame):
    """Create a color selection area on the left side of the screen."""
    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'eraser': (255, 255, 255) 
    }
    button_list = []
    for idx, (color_name, color_value) in enumerate(colors.items()):
        x, y = 10, 100 + idx * 60
        w, h = 50, 50
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_value, -1)
        button_list.append((x, y, w, h, color_value))
    return button_list

def check_color_selection(x, y, button_list):
    """Check if the thumb is selecting a color from the palette."""
    for button in button_list:
        bx, by, bw, bh, color = button
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return color
    return None

def is_index_finger_up(hand_landmarks):
    """Check if the index finger is up for writing or selecting colors."""
    return hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < \
           hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y

def main():
    args = get_args()

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    current_color = (255, 0, 0)  # Default: Red
    index_finger_up_frames = 0  

    shape = generate_random_shape(args.width, args.height)  # Generate one random shape
    traced_points = set()  

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        button_list = create_color_buttons(frame)
        draw_shape(frame, shape)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                thumb_x, thumb_y = int(thumb_tip.x * args.width), int(thumb_tip.y * args.height)

                selected_color = check_color_selection(thumb_x, thumb_y, button_list)
                if selected_color:
                    current_color = selected_color  # Ensure correct color selection

                index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y = int(index_tip.x * args.width), int(index_tip.y * args.height)

                if is_index_finger_up(hand_landmarks):
                    index_finger_up_frames += 1
                else:
                    index_finger_up_frames = 0  

                if index_finger_up_frames > 20:  # Only trace when index is consistently up
                    if current_color == (255, 255, 255):  
                        cv2.circle(canvas, (index_x, index_y), 20, (0, 0, 0), -1)  
                    else:
                        cv2.circle(canvas, (index_x, index_y), 10, current_color, -1)
                        traced_points.add((index_x, index_y))  

        cv2.putText(frame, 'Trace the shape using the correct color!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow('Gesture Writing - Trace the Shape', frame)

        if is_tracing_correct(traced_points, shape, current_color):  
            cv2.putText(frame, "Well Done!", (args.width // 3, args.height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Gesture Writing - Trace the Shape', frame)
            cv2.waitKey(2000)  
            break  

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
