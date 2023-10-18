# Ball Center and Curve Fitting

This Python script is designed to perform two main tasks: locating the center of a ball in a series of images and fitting a curve to the ball's trajectory. It can be useful for analyzing and tracking the motion of objects like a ball in a video or image sequence.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Tasks](#4-tasks)
5. [Visualization](#5-visualization)

## 1. Introduction

The script performs two primary tasks:

1. **Ball Center Detection**: It detects the center of a red ball in a series of images or frames captured from a video.

2. **Curve Fitting**: It fits a curve to the detected ball's trajectory using a quadratic polynomial.

## 2. Installation

Before using this script, you need to install the required dependencies. You can install them using the following command:

```bash
pip install cv2 numpy matplotlib
```

## 3. Usage
To use this script, follow these steps:

Provide the path to a video file containing the ball's motion. The script reads the video frame by frame.
Ensure you have an image (e.g., 'download.png') for detecting the red ball.
Adjust the red ball's color range by modifying the lower_red and upper_red values to match the ball's color in your images.
Run the script. It processes the video frames, detects the ball's center, and fits a curve to its trajectory.
The script will display a scatter plot of the ball's trajectory and the fitted curve.

## 4. Tasks
Ball Center Detection
The script detects the center of a red ball in each frame using the Ball_Centre function. It calculates the x and y coordinates of the ball's center and appends these coordinates to the coordinates list.

Curve Fitting
The script fits a quadratic polynomial curve to the ball's trajectory. It uses the fit_curve function to find the best-fitting curve and displays the fitted curve in green on the scatter plot.

## 5. Visualization
The script provides a visualization of the detected ball's trajectory and the fitted curve. You can observe the motion of the ball and analyze its path.

Please feel free to adapt and modify the script to suit your specific use case or project. Enjoy tracking and analyzing the motion of objects using this script!

Happy coding!


