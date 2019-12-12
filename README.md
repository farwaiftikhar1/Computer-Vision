# Computer-Vision
Lane Line Detection Algorithm for Autonomous Vehicles
Abstract
The objective of this project was to implement a lane detection algorithm to detect straight and curved yellow and white lanes on the road using computer vision techniques implemented through OpenCV Python.
Introduction
One of the tasks performed by human drivers is to identify lanes on the road and stay in them for smooth driving and avoiding collisions. Among one of the daunting tasks for the future road vehicles especially autonomous road vehicles is road lane detection or road boundaries detection. A robust and reliable algorithm is a minimum requirement as erroneous findings could generate wrong steering commands jeopardizing vehicle’s safety.
The state-of-the-art method to detect road lane lines is to use vision system on the vehicle. In this project well-developed computer vision techniques are employed for detection of lane lines in real time with robustness and accuracy. The front view of the road is acquired using a camera mounted on the vehicle and the input is processed using the below mentioned pipeline.
Project Pipeline:
•	Computation of camera calibration matrix and distortion coefficients from a set of chessboard images
•	Distortion removal on images
•	Application of gradient thresholds to find edges
•	Application of Hough Transform to generate lane lines
•	Perform perspective transform to generate bird’s eye view
•	Use of sliding windows to find hot lane line pixels
•	Fitting of second degree polynomials to identify left and right lane curves composing the lane
•	Displaying the lane lines on the image

