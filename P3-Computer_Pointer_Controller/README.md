# Computer Pointer Controller



It's an application that uses a gaze detection model to control the mouse pointer of your computer. 

## Project Set Up and Installation

[TOC]

## Setup

1-Download and install [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

2-Download Models Inference Files using openVINO model downloader.

1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
3. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

3- Install the dependencies.

```
pip3 install -r requirements.txt
```

### Application pipeline

![](C:\Users\007\Desktop\pipeline.png)

### Application Tree

```
│   .Instructions.md.swp
│   main.py
│   README.md
│   requirements.txt
│
├───bin
│       demo.mp4
│
└───src
    │   face_detection.py
    │   facial_landmarks_detection.py
    │   gaze_estimation.py
    │   head_pose_estimation.py
    │   input_feeder.py
    │   mouse_controller.py
    │
```

- **src** folder contains python files of the app
  - **face_detection.py** : Face Detection related inference code
  - **facial_landmarks_detection.py** : Landmark Detection related inference code
  - **gaze_estimation.py** : Gaze Estimation related inference code
  - **head_pose_estimation.py** : Head Pose Estimation related inference code
  - **input_feeder.py** : input selection 
  - **mouse_controller.py** : Mouse Control related utilities.
- **bin** folder contains the media 
- **main.py** : Main code to run the app
- **README.md** : contains the instructions   

## Demo

**1**-Create a python  Virtual env.

```
# Create a Virtual env
python3 -m venv openvino-env
```

**2**- Install requirements  

```pip3 install -r requirements.txt```

**3**- Run  ***setupvars***

- ***windows*** ```cd C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\bin``` ***then*** ```setupvars.bat``
- ***ubuntu***  ```bash /opt/intel/openvino/bin/setupvars.sh```

4- In project directory run the following code (minimum requirements ) after writing the paths of models and the path or type of input

```
python3 main.py -mfd <PATH OF FACE_DETECTION MODEL> -mlm <PATH OF LANDMARKS_REGRESSION_RETAIL MODELl> -mhp <PATH OF HEAD_POSE_ESTIMATION_ADAS> -mge<PATH OF GAZE_ESTIMATION_ADAS MODEL> -i <PATH OF INPUT_VIDEO >  
```

![](C:\Users\007\Desktop\Mc.png)

![](C:\Users\007\Desktop\mc2.png)

## Documentation
```
usage: main.py [-h] -mfd FDMODEL -mhp HPMODEL -mlm LMMODEL -mge GEMODEL -i
               INPUT [-d DEVICE] [-l CPU_EXTENSION] [-pt PROB_THRESHOLD]
               [-dp DISPLAY_OUTPUTS] [-bmfd BENCHMARK_FACE_DETECTION]
               [-bmfl BENCHMARK_FACIAL_LANDMARK_DETECTION]
               [-bmhp BENCHMARK_HEAD_POSE_ESTIMATION]
               [-bmge BENCHMARK_GAZE_ESTIMATION]

optional arguments:
  -h, --help            show this help message and exit
  -mfd FDMODEL, --fdmodel FDMODEL
                        Path to a face detection xml file with a trained
                        model.
  -mhp HPMODEL, --hpmodel HPMODEL
                        Path to a head pose estimation xml file with a trained
                        model.
  -mlm LMMODEL, --lmmodel LMMODEL
                        Path to a facial landmarks xml file with a trained
                        model.
  -mge GEMODEL, --gemodel GEMODEL
                        Path to a gaze estimation xml file with a trained
                        model.
  -i INPUT, --input INPUT
                        Path video file or CAM to use camera
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.6 by
                        default)
  -dp DISPLAY_OUTPUTS, --display_outputs DISPLAY_OUTPUTS
                        Display the outputs of the models.
  -bmfd BENCHMARK_FACE_DETECTION, --benchmark_face_detection BENCHMARK_FACE_DETECTION
                        Print the time it takes for each layer in
                        face_detection model
  -bmfl BENCHMARK_FACIAL_LANDMARK_DETECTION, --benchmark_facial_landmark_detection BENCHMARK_FACIAL_LANDMARK_DETECTION
                        Print the time it takes for each layer in
                        facial_landmark_detection model
  -bmhp BENCHMARK_HEAD_POSE_ESTIMATION, --benchmark_head_pose_estimation BENCHMARK_HEAD_POSE_ESTIMATION
                        Print the time it takes for each layer in
                        head_pose_estimation model
  -bmge BENCHMARK_GAZE_ESTIMATION, --benchmark_gaze_estimation BENCHMARK_GAZE_ESTIMATION
                        Print the time it takes for each layer in
                        gaze_estimation model
```

## Benchmarks

For Device = **CPU**

| Model Type               | Face Detection | Facial Landmarks Detection | Head pose Estimation | Gaze Estimation |
| ------------------------ | -------------- | -------------------------- | -------------------- | --------------- |
| load time FP32           | 224 ms         | 96 ms                      | 92 ms                | 121 ms          |
| load time FP16-INT8      | NA             | 139 ms                     | 103 ms               | 169 ms          |
| load time FP16           | NA             | 117 ms                     | 105 ms               | 138 ms          |
| Inference Time FP32      | 10 ms          | 0.5 ms                     | 0.9 ms               | 1 ms            |
| Inference Time FP16-INT8 | NA             | 0.3 ms                     | 0.6 ms               | 0.7 ms          |
| Inference Time FP16      | NA             | 0.4 ms                     | 1 ms                 | 1 ms            |

For Device = **IGPU**

| Model Type               | Face Detection | Facial Landmarks Detection | Head pose Estimation | Gaze Estimation |
| :----------------------- | -------------- | -------------------------- | -------------------- | --------------- |
| load time FP32           | 225 ms         | 94  ms                     | 98 ms                | 123 ms          |
| load time FP16-INT8      | NA             | 104 ms                     | 137 ms               | 167 ms          |
| load time FP16           | NA             | 95 ms                      | 104 ms               | 127 ms          |
| Inference Time FP32      | 10 ms          | 0.4 ms                     | 0.9 ms               | 1 ms            |
| Inference Time FP16-INT8 | NA             | 0.5 ms                     | 0.8 ms               | 0.8 ms          |
| Inference Time FP16      | NA             | 0.3 ms                     | 1 ms                 | 1 ms            |

**Models' size**

| Model Type | Face Detection | Facial Landmarks Detection | Head pose Estimation | Gaze Estimation |
| ---------- | -------------- | -------------------------- | -------------------- | --------------- |
| FP32       | 1.86 MB        | 0.786 MB                   | 7.34 MB              | 7.24 MB         |
| FP16-INT8  | NA             | 0.314 MB                   | 2.05 MB              | 2.09 MB         |
| FP16       | NA             | 0.413 MB                   | 3.69 MB              | 3.65 MB         |



## Results
- There is no big difference between **CPU** and **IGPU** , they give almost the sane result.

- ```FP32```models have the lowest load time but have a high inference time and size comparing to other types.
- ```FP16-INT8```  models has the lowest inference time for **CPU**  



## Stand Out Suggestions
Model size can reduce by lowing the precision from FP32 to FP16 or INT8 and inference becomes faster but because of lowing the precision model can lose some of the important information because of that accuracy of model can decrease.

It's recommended to use  **FP16-INT8** models as it have a low inference time and the lowest size

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow.

- If there is more than one face detected, it extracts only one face and do inference on it and ignoring other faces.
- Moving the mouse pointer out of the maximum window width, will finished the app.