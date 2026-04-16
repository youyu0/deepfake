# deepfake

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
- Support a single video file or a folder of videos
- Save processed output videos with annotations
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
├── detect_from_video.py
├── models.py
├── transform.py
├── xception.py
├── requirement.txt
└── README.md
```

## How It Works

1. Read frames from the input video
2. Detect faces using dlib
3. Crop the detected face region
4. Preprocess the cropped face image
5. Run inference with the Xception model
6. Draw prediction results on the frame
7. Save the processed output video

## Installation

Clone the repository:

```bash
git clone https://github.com/youyu0/deepfake.git
cd deepfake
```

Install dependencies:

```bash
pip install -r requirement.txt
```

## Usage

Run on a single video:

```bash
python detect_from_video.py -i input.mp4 -mi model.pth -o output_folder --cuda
```

Run on a folder of videos:

```bash
python detect_from_video.py -i ./videos -mi model.pth -o ./results --cuda
```

## Arguments

- `-i`, `--video_path`  
  Path to a video file or a folder containing videos

- `-mi`, `--model_path`  
  Path to the trained model checkpoint

- `-o`, `--output_path`  
  Directory used to save processed videos

- `--start_frame`  
  Start frame index

- `--end_frame`  
  End frame index

- `--cuda`  
  Enable CUDA inference

## Output

The processed video will contain:

- detected face bounding boxes
- prediction confidence scores
- classification result (`real` or `fake`)

## Notes

- This repository mainly focuses on **video inference**
- A trained model checkpoint is required before running detection
- Some dependencies such as `dlib` may require extra setup depending on your operating system
- You may need to adjust model weight paths for your local environment

## Limitations

- The model only classifies the detected face region
- Performance depends on the quality of the trained weights
- Environment setup may vary across different systems

## Future Improvements

- Add training pipeline and dataset instructions
- Support multiple face detections per frame
- Add sample videos and demo results
- Improve checkpoint loading and configuration
- Add Docker or environment setup guide

## Author

youyu0

## License

This repository currently does not specify a license.
