# Object detection and segmentation with Mask R-CNN

## About The Project:

This project is built using PyTorch implementation of Mask R-CNN and Python 3.11.9

## Usage:

+ Clone the GitHub repository and run the following command to initialize it
```
get init
```
+ Create a virtual environment and install requirements (packages) using the following command
 ```
 pip install -r requirements.txt
 ```
+ Modify the 'input_video_path' variable  in 'mask r-cnn.py' file by inserting the path to your input video

<!-- BEGIN_Specification: -->
## Specification:

### Inputs
| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_video_path"></a> [input\_video\_path](#input\_video\_path) | path to the input video | `string` | n/a | yes |

### Outputs

| Name | Description |
|------|-------------|
| <a name="mrcnn_annotations.json"></a> [mrcnn\_annotations.json](#mrcnn_annotations.json) | JSON file with the model predictions  |
| <a name="output_video.mp4"></a> [output\_video.mp4](#output_video.mp4) | output video with the model predictions  |
<!-- END_Specification: -->
