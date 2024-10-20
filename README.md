# Harmful Object Detection App

## Overview

The Harmful Object Detection App is a web application developed using Streamlit and YOLO (You Only Look Once) to detect harmful objects such as knives, guns, and scissors in images, videos, or through webcam input. The application utilizes deep learning for real-time object detection and plays a siren sound when harmful objects are detected.

## Features

- Upload images or videos for detection.
- Real-time detection using the webcam.
- Siren sound playback upon detection of harmful objects.
- Display of detected objects on images and video frames.

## Installation

### Prerequisites

Make sure you have Python 3.7 or higher installed on your system.

It's recommended to create a virtual environment to manage your dependencies. You can do this using `venv`:

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
