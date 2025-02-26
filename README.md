Gesture Revolution: Rehabilitation System

Project Overview
This project introduces a gesture-based rehabilitation system designed to support stroke survivors and individuals with motor impairments by using hand-tracking technology. The system integrates gamified exercises that promote fine motor skill recovery in an interactive and engaging manner.

The project leverages OpenCV, MediaPipe, and NumPy to process real-time hand tracking, allowing users to interact with the exercises using simple hand gestures. The system is composed of four core exercises, each targeting a specific aspect of motor function.

Features:
🎯 Ball Catch Exercise – Improves reaction time and grip strength.
✍️ Gesture Tracing Exercise – Enhances fine motor precision and control.
🖐 Grip Count Exercise – Helps build grip endurance and strength.
➰ Line Control Exercise – Develops movement accuracy and stability.

Installation
To run this project locally, ensure you have Python 3.8+ installed.

Clone this repository:
git clone https://github.com/SaadASyed/Gesture-Revolution.git
cd Gesture-Revolution

Install required dependencies:
pip install -r requirements.txt

Run the desired exercise:
python src/1. Ball_catch.py

Usage:
Each exercise functions as an independent module, allowing users to train specific motor skills. Below is a brief guide on running each exercise:

1️⃣ Ball Catch Exercise
The user sets a target number of successful catches.
A virtual ball appears, and the user must make a gripping motion at the right time.
A counter tracks successful catches, and a congratulatory message appears upon completion.

2️⃣ Gesture Tracing Exercise
A shape (circle, square, or triangle) appears randomly.
The user must trace the shape with their index finger using the correct color.
The system validates completion only when the shape is fully outlined.

3️⃣ Grip Count Exercise
The system detects gripping motions in real-time.
The user sets a goal for the number of grips to complete.
A counter keeps track of successful grips, and feedback is provided.

4️⃣ Line Control Exercise
A jagged line appears across the screen.
The user guides their index finger along the path while avoiding deviations.
A counter records mistakes, and performance feedback is given.