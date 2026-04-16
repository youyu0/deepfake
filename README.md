# Deepfake Detection from Video

A Python-based deepfake detection project that detects faces in video frames and classifies them as **real** or **fake** using an **Xception-based binary classification model**.

## Overview

This project processes a video file frame by frame, detects human faces using **dlib**, crops the detected face region, and feeds it into a trained deep learning model for deepfake classification.

The output video is saved with:
- face bounding boxes
- prediction scores
- final labels (`real` / `fake`)

## Features

- Detect faces from video frames
- Classify each detected face as real or fake
- Support single video file or a folder of videos
- Save processed output video with annotations
- Built with PyTorch and OpenCV

## Tech Stack

- Python
- PyTorch
- OpenCV
- dlib
- Pillow
- torchvision
- pretrainedmodels
- tqdm

## Project Structure

```bash
deepfake/
├── detect_from_video.py   # Main script for video-based deepfake detection
├── models.py              # Model selection and transfer learning setup
├── transform.py           # Image preprocessing / normalization pipeline
├── xception.py            # Xception model definition
├── requirement.txt        # Project dependencies
└── README.md
