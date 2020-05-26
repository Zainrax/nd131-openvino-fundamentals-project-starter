## Introduction

This reports aims to clarify the components of the provided code, as well as give my personal findings in OpenVINO. The application is able to accurately predict the number of people within a frame, the total amount of people in a given input, and the duration a person stays. The system I built is robust enough to handle fluctuations in the model inference, however further testing would need to be done to handle novel tasks such as if two people are stack on top of each other in a frame. Due to a lack of domain knowledge, much of the parsing code for the Yolov3 model output is credited to this [source](https://github.com/PINTO0309/OpenVINO-YoloV3) though much of the code has been heavily modified to fit the current task.

## Explaining Custom Layers

The process behind converting custom layers depends on the framework one uses for a given model. The model optimiser looks at the different layers of providing optimisation for the known layers, and classifies the unknown types of layers as custom layers. Given that we have some custom layers we are able to provide an extension to the model optimiser to account for them. The Caffe framework allows for the calculation of output shape of a layer. Tensorflow models have the option to have a sub-graph replacement in which the model optimiser performs the replacement steps.

The ability to add custom layers is key when making our models as it removes the limitation if someone wanted to create their own activation function that would otherwise be unrecognised by the the model optimiser

## Comparing Model Performance

The original YoloV3 model was trained using the coco dataset with the intermediate representation(IR) using an Tensorflow implementation of yolov3, with model weights based on the same coco dataset. The IR aims to reduce the time for inference on an input, reduce the overall size of the model, while limiting the impact on accuracy. I am unable to comment on the exact impact of accuracy as I am unable to accurately compare the original model for use with the reference video provided, however, I was able to build comprehensive understanding around the difference in model size and inference time. The IR remained similar in size with 236 MB for the bin file compared to the Tensorflow .pb with 237 MB. Though the size remained similar our inference was on average 12 msec/frame, which compared to the results found in the original paper ([source](Redmon, J., & Farhadi, A. (2018) Yolov3: An incremental improvement._arXiv preprint arXiv:1804.02767_.) ) on the same input size had an average 29 msec for an inference. While the result can not be stated as a direct comparison it is clear the optimisations and inference network place a large role in improving the inference speed of a model.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- Social Distance Grouping: given the current world situation it can gives a sense of how many people are in a space, allowing us to determine if it has reached a threshold.
- Surveying: looking at footpath traffic and density around certain areas and how that can change over time.
- Shopping Centres:  can be used to density around certain sections of a store giving a sense of what people typically look at or where they gather.

Each of these use cases would be useful because they give a sense of how a space is used. This relies on a distributed systems to compare, and contrast across several spaces but the result is an insight on how people move, how long they stay, and the overall density of a given area.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are the observed result at any given frame is not sufficient in making a assumption of what is actually there. The model by itself would be inefficient by itself without certain parsing and state management across a data source(at least in the case of video). The blips that occur due to bad angles, poor lighting, etc. dramatically throw the given numbers, such it's important to have a parsing system that is able to account for these occurrences. The end user if given the raw model output would have a large amount of inaccurate, unusable data that would not be useful to make any sort of claims upon.

## Conclusion

The end result is a system that is able to quickly and accurately give information about people with a space given certain functionality that provides a robust system of inference. The end result is a program that I can see a lot of possible potential for, and would look to see it's effects in a real edge environment.
