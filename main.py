"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = "localhost"
IPADDRESS = socket.gethostbyaddr(HOSTNAME)
MQTT_HOST = "localhost"
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

labels = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
          "truck", "boat", "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
          "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
          "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
          "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
          "donut", "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush")

classes = 80
coords = 4


def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) +
               entry * side * side + loc)


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(
        box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(
        box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


class DetectionObservation():
    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.confidence = confidence
        self.class_id = class_id


def parseYoloV3(out, threshold):
    observations = []
    side = out.shape[2]
    side_sq = side * side

    out_blob = out.flatten()
    for i in range(side_sq):
        for n in range(3):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i,
                                   coords)
            scale = out_blob[obj_index]
            if (scale < threshold):
                continue
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes,
                                         n * side_sq + i, coords + 1 + j)
                prob = scale * out_blob[class_index]
                if prob < threshold:
                    continue
                observation = DetectionObservation(j, prob)
                observations.append(observation)

    return observations


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m",
                        "--model",
                        required=True,
                        type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        type=str,
                        help="Path to image or video file")
    parser.add_argument("-l",
                        "--cpu_extension",
                        required=False,
                        type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                        "Absolute path to a shared library with the"
                        "kernels impl.")
    parser.add_argument("-d",
                        "--device",
                        type=str,
                        default="CPU",
                        help="Specify the target device to infer on: "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                        "will look for a suitable plugin for device "
                        "specified (CPU by default)")
    parser.add_argument("-pt",
                        "--prob_threshold",
                        type=float,
                        default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model_xml=args.model,
                             cpu_ext=args.cpu_extension,
                             device=args.device)

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    cam_h = 768
    cam_w = 432
    new_w = int(cam_w * 416 / cam_w)
    new_h = int(cam_h * 416 / cam_h)

    # get frame inforamtion
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(duration)

    input_shape = infer_network.get_input_shape()
    width = input_shape[2]
    height = input_shape[3]

    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        flag, frame = cap.read()
        curr_time = frame_count / fps
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        p_frame = cv2.resize(frame, (width, height))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        infer_network.exec_net(0, p_frame)
        if infer_network.wait() == 0:
            outputs = infer_network.get_output()
            #count = get_people_count(result, args.prob_threshold)
            for result in outputs:
                observations = parseYoloV3(result, prob_threshold)
            for obs in observations:
                print(obs.class_id)
                print(obs.confidence)
    ### TODO: Handle the input stream ###

    ### TODO: Loop until stream is over ###

    ### TODO: Read from the video capture ###

    ### TODO: Pre-process the image as needed ###

    ### TODO: Start asynchronous inference for specified request ###

    ### TODO: Wait for the result ###

    ### TODO: Get the results of the inference request ###

    ### TODO: Extract any desired stats from the results ###

    ### TODO: Calculate and send relevant information on ###
    ### current_count, total_count and duration to the MQTT server ###
    ### Topic "person": keys of "count" and "total" ###
    ### Topic "person/duration": key of "duration" ###

    ### TODO: Send the frame to the FFMPEG server ###

    ### TODO: Write an output image if `single_image_mode` ###


def get_people_count(result, threshold):
    count = 0
    layer = result[0]

    for detection in layer:
        for idx, s in enumerate(detection):
            if s[5] > threshold:
                print("-{}-".format(idx))
                print(s[5])

    return count


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
