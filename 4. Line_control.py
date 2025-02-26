#Programmer: Saad Syed
#Code description: code for line control exercise
#Last update date: 5/2/25
#Update description: updated code to ensure change of colour when circle is off the line

import cv2 
import mediapipe as mp
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()

def draw_full_squiggly_line(img, height, width):
    """Draws a **continuous** jagged line from one end of the screen to the other."""
    mid_y = height // 2  # Center Y position

    # Ensure the line spans **exactly from left to right edge**
    points = [(0, mid_y + np.random.randint(-40, 40))]
    for i in range(1, 20):  # Generates 20 segments
        x = int(i * (width / 19))  # Spaced evenly across the width
        y = mid_y + np.random.randint(-40, 40)  # Adds variation
        points.append((x, y))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (0, 0, 255), 2)  # Red jagged line
    
    return np.array(points, dtype=np.int32)

def distance_to_line_segment(pt, line_start, line_end):
    """Computes the shortest distance from a point to a line segment."""
    line_vec = np.array(line_end) - np.array(line_start)
    pt_vec = np.array(pt) - np.array(line_start)
    line_length = np.dot(line_vec, line_vec)
    
    if line_length == 0:
        return np.linalg.norm(pt_vec)
    
    projection = np.dot(pt_vec, line_vec) / line_length
    projection = max(0, min(1, projection))
    closest_point = np.array(line_start) + projection * line_vec
    return np.linalg.norm(np.array(pt) - closest_point)

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

    path_points = draw_full_squiggly_line(np.zeros((args.height, args.width, 3), dtype=np.uint8), args.height, args.width)  
    ring_radius = 20
    out_of_bounds_count = 0  # Counter for off-path movements
    prev_collision_status = True  # Start assuming index is on the line

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw the **continuous** jagged line
        for i in range(len(path_points) - 1):
            cv2.line(frame, tuple(path_points[i]), tuple(path_points[i+1]), (0, 0, 255), 2)  # Red jagged line

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pos = (int(index_tip.x * args.width), int(index_tip.y * args.height))

                # Collision detection: Check if index finger is close to the jagged line
                collision_detected = any(
                    distance_to_line_segment(index_pos, path_points[i], path_points[i+1]) < ring_radius
                    for i in range(len(path_points) - 1)
                )

                # Track instances when the circle moves off the red line
                if not collision_detected and prev_collision_status:
                    out_of_bounds_count += 1  

                prev_collision_status = collision_detected  # Update status

                # Change color: Green when on the line, Red when off
                ring_color = (0, 255, 0) if collision_detected else (0, 0, 255)
                cv2.circle(frame, index_pos, ring_radius, ring_color, 2)  # Draw index finger tracking circle

        # Display counter on screen
        cv2.putText(frame, f'Mistakes: {out_of_bounds_count}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Line Control Exercise', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
