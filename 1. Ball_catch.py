#Programmer: Saad Syed
#Code description: code for the ball catch exercise
#Last update date: 5/2/25
#Update description: Counter incorporated

import cv2
import mediapipe as mp
import numpy as np
import random

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()

def generate_random_position(width, height, margin=100):
    """Generates a random position for the ball, keeping it within a margin from the edges."""
    x = random.randint(margin, width - margin)
    y = random.randint(margin, height - margin)
    return (x, y)

def is_fist_clenched(hand_landmarks):
    """Checks if all fingers are retracted, indicating a clenched fist."""
    mp_hands = mp.solutions.hands
    fingers_retracted = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ]
    return all(fingers_retracted)

def get_num_balls_from_screen(cap, width, height):
    """Allows the user to enter the number of balls via keyboard input on the game screen."""
    input_value = ""
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display instruction
        cv2.putText(frame, "Enter the number of balls to catch:", (width // 6, height // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display current input
        cv2.putText(frame, input_value, (width // 2, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('Catch the Moving Ball', frame)

        key = cv2.waitKey(0) & 0xFF  # Wait for key press

        if key == 13:  # Enter key
            if input_value.isdigit():
                return int(input_value)  # Convert input to integer and return
        elif key == 8:  # Backspace key
            input_value = input_value[:-1]  # Remove last character
        elif key in range(48, 58):  # Number keys (0-9)
            input_value += chr(key)  # Append the typed number to input

def main():
    args = get_args()
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Get number of balls from on-screen input
    num_balls = get_num_balls_from_screen(cap, args.width, args.height)
    balls_caught = 0  # Counter for caught balls

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    ball_position = generate_random_position(args.width, args.height, margin=100)
    ball_radius = 30

    while balls_caught < num_balls:  # Continue until the target is reached
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw the ball on the screen
        cv2.circle(frame, ball_position, ball_radius, (0, 0, 255), -1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                palm_x, palm_y = int(palm_base.x * args.width), int(palm_base.y * args.height)

                cv2.circle(frame, (palm_x, palm_y), 20, (255, 0, 0), 2)

                distance = np.sqrt((palm_x - ball_position[0]) ** 2 + (palm_y - ball_position[1]) ** 2)
                if distance < ball_radius + 20 and is_fist_clenched(hand_landmarks):  
                    balls_caught += 1
                    ball_position = generate_random_position(args.width, args.height, margin=100)

        # Display score
        cv2.putText(frame, f'Balls Caught: {balls_caught}/{num_balls}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Catch the Moving Ball', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Display congratulatory message on the game screen
    for _ in range(100):  # Display message for a few seconds
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, "Congratulations!", (args.width // 3, args.height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.putText(frame, f"You caught {num_balls} balls!", (args.width // 4, args.height // 2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow('Catch the Moving Ball', frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
