## Introduction

This reports aims to clarify the components of the provided code, as well as give my personal findings in OpenVINO. The application is able to accurately predict the number of people within a frame, the total amount of people in a given input, and the duration a person stays. The system I built is robust enough to handle fluctuations in the model inference, however further testing would need to be done to handle novel tasks such as if two people are stack on top of each other in a frame. The initial model was done using multi versions of Yolo object detection however I was unsucessful in finding a usuable version though the results were accurate, the inference after parsing the result was far too slow ranging from 0.2 using Yolov3-tiny, to up to 1 second per frame. I thus relied on the pre-trained model [person-detection-retail-0013](https://docs.openvinotoolkit.org/2019_R1/_person_detection_retail_0013_description_person_detection_retail_0013.html), though I detail the process

## Explaining Custom Layers

The process behind converting custom layers depends on the framework one uses for a given model. 
The model optimiser creates a network topology of the different layers of a given model providing optimisation for the known layers, and classifing the unknown layer types as custom layers. Given that we have some custom layers we are able to provide an extension to the model optimiser to account for them. The Caffe framework allows for the calculation of output shape of a layer. Tensorflow models have the option to have a sub-graph replacement in which the model optimiser performs the replacement steps. Each of the different frameworks have different features that may need to be accounted for, and custom layers are there to help the model optimizer to understand the network for futher optimisation.

The ability to add custom layers is key when making our models as it removes the limitation if someone wanted to create their own activation function that would otherwise be unrecognised by the the model optimiser.

## Comparing Model Performance

The original YoloV3 model was trained using the coco dataset with the intermediate representation(IR) using an Tensorflow implementation of yolov3, with model weights based on the same coco dataset. The IR aims to reduce the time for inference on an input, reduce the overall size of the model, while limiting the impact on accuracy. I am unable to comment on the exact impact of accuracy as I am unable to accurately compare the original model for use with the reference video provided, however, I was able to build comprehensive understanding around the difference in model size and inference time. The IR remained similar in size with 236 MB for the bin file compared to the Tensorflow .pb with 237 MB. Though the size remained similar our inference was on average 12 msec/frame, which compared to the results found in the original paper ([source](Redmon, J., & Farhadi, A. (2018) Yolov3: An incremental improvement._arXiv preprint arXiv:1804.02767_.) ) on the same input size had an average 29 msec for an inference. While the result can not be stated as a direct comparison it is clear the optimisations and inference network place a large role in improving the inference speed of a model.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

- Social Distance Grouping: given the current world situation it can gives a sense of how many people are in a space, allowing us to determine if it has reached a threshold.
- Surveying: looking at footpath traffic and density around certain areas and how that can change over time.
- Shopping Centres:  can be used to density around certain sections of a store giving a sense of what people typically look at or where they gather.

Each of these use cases would be useful because they give a sense of how a space is used. This relies on a distributed systems to compare, and contrast across several spaces but the result is an insight on how people move, how long they stay, and the overall density of a given area.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are the observed result at any given frame is not sufficient in making a assumption of what is actually there. The model would be inefficient by itself without certain parsing and state management across a data source(at least in the case of video). The blips that occur due to bad angles, poor lighting, etc. dramatically throw the given numbers, as such it's important to have a parsing system that is able to account for these occurrences. The end user if given the raw model output would have a large amount of inaccurate, unusable data that would not be useful to make any sort of claims upon.

## Models Used
The original three models used were YoloV3, YoloV4, YoloV3-tiny. All three were based on the [Converting Yolo models to IR](https://docs.openvinotoolkit.org/2020.1/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)
**YoloV3 & YoloV3-Tiny**
File Download:

```
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
git checkout ed60b90
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

Pb Conversion:
```
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny

```

IR Conversion:
```
python3 mo_tf.py
--input_model /path/to/yolo_v3.pb
--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
--batch 1
python3 mo_tf.py
--input_model /path/to/yolo_v3_tiny.pb
--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3_tiny.json
--batch 1
```
**YoloV4**
Used a yolov4-tflite version [repo](https://github.com/hunglc007/tensorflow-yolov4-tflite) with weights from [here](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT).

```
python convert.py --weights ./data/yolov4.weights --output ./data/yolov4-pb
python3 mo_tf.py
--input_model /path/to/yolo_v4.pb
--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
--batch 1
```
While I was able to get a IR representation, the output was null, even after matching the anchors.
The YoloV3 ended up having a 1 second inference per frame which may be due the method for parsing, but the accuracy is viable. The Tiny had 0.2 second inference which is much faster, but the accuracy went from an average of 0.8/9 to 0.4~ making it unreliable. The yolov4 outputted empty arrays due to incompatability of the conversion.

## Model Solution
This  model person-detection-retail-0013 was a first choice as a backup due to it's comparatively high 88% AP vs the alternative person-detection-retail-0002, which was also larger. The IR model can be downloaded using the downloader.py
**Download**
```<OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP16```

## Conclusion

While there was a number of issues around the inference times of Yolo, it's clear that it was able to increase the inference time by a larger factor, and with careful coding around the lapses or inaccuracies in the model,
the tradeoffs are slim.
