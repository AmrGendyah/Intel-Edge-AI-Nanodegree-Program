# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves model optimization to extract concerned layers to use with concerned hardware (CPU, GPU...), and running the conversion by specifying layers implementations and adding extensions to the Model Optimizer and the Inference Engine

Some of the potential reasons for handling custom layers are
- using an unsupported layer by the model optimizer.

## Comparing Model Performance
### ssd_mobilenet_v2
The mAP is 22
The inference time of the model pre-conversion was 31ms and post-conversion was 70 ms

### ssd_resnet50_v1
The mAP is 35
The inference time of the model pre-conversion was 76ms and post-conversion was 2652 ms

### ssdlite_mobilenet_v2_coco
The mAp is 22
The inference time of the model pre-conversion was 27ms and post-conversion was 31 ms

### person-detection-retail-0013
The inference time of the model is 44ms and it counts 11 persons

### pedestrian-detection-adas-0002
The inference time of the model is 55ms and it counts 36 persons

## Assess Model Use Cases

Some of the potential use cases of the people counter app are :
1-Movie theater
2-Cafe
3-Class
4-Hospital
5-Office

Each of these use cases would be useful because
1- Making a big class for students
2- Expansion for a hospital or a cafe
3- Maneging the working time to avoid the crowding in a office

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows
- A bad lighting can decrease model performance by diffusing image info.
- Physical defects. Surface defects will cause light rays to focus at a different point leading to blurring.
- Non-uniform illumination and shading across the image. With any lens, image brightness is reduced towards the edges and this is known as vignetting. Cos4 vignetting occurs because the light has to travel further to the edge of the image and reaches the sensor at a shallower angle. Mechanical vignetting occurs when the light beam is mechanically blocked, usually by the lens mount.
- Changing focal length of a lens magnifies or reduces the size of an image compared to the size of an image formed by an intermediate focal length lens.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_mobilenet_v2]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments ``` python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json ```
  
  - The model was insufficient for the app because it counts 36 persons
 
- Model 2: [ssd_resnet50_v1]
  - [http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
  - The model was insufficient for the app because it takes huge amount of time to analyze video or frames and the inference time is 2652 ms
  

- Model 3: [ssdlite_mobilenet_v2]
  - [http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
  - The model was insufficient for the app because it counts 42 persons 
  
  
  ## Run the Code
  The used model is *person-detection-retail-0013* 
  
  ```python main.py -i "resources/Pedestrian_Detect_2_1_1.mp4" -m "intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml" -l "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so" -d "CPU" -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm```
