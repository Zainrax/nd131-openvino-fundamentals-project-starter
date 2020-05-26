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
import math
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

anchors = [
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373,
    326
]
classes = 80
coords = 4


class DetectionObservation():
    time_found = 0.0
    last_updated = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, w_scale, h_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.confidence = confidence
        self.class_id = class_id


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


def parseYoloV3(out, threshold, cap_h, cap_w):
    observations = []
    side = out.shape[2]
    side_sq = side * side

    offset = 0
    if side == 13:
        offset = 2 * 6
    if side == 26:
        offset = 2 * 3
    if side == 52:
        offset = 2 * 0

    out_blob = out.flatten()
    for i in range(side_sq):
        row = int(i / side)
        col = int(i % side)
        for n in range(3):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i,
                                   coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i,
                                   0)
            scale = out_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + out_blob[box_index + 0 * side_sq]) / side * 416
            y = (row + out_blob[box_index + 1 * side]) / side * 416
            height = math.exp(out_blob[box_index + 3 *
                                       side_sq]) * anchors[offset + 2 * n + 1]
            width = math.exp(
                out_blob[box_index + 2 * side_sq]) * anchors[offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes,
                                         n * side_sq + i, coords + 1 + j)
                prob = scale * out_blob[class_index]

                if prob < threshold:
                    continue
                observation = DetectionObservation(x, y, height, width, j,
                                                   prob, (cap_h / 416),
                                                   (cap_w / 416))

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
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get frame inforamtion
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    input_shape = infer_network.get_input_shape()
    width = input_shape[2]
    height = input_shape[3]

    frame_count = 0
    people_count = 0
    found_people = []
    total_inference_time = []
    while cap.isOpened():
        people_in_frame = 0
        frame_count += 1
        flag, frame = cap.read()
        curr_time = frame_count / fps
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        p_frame = cv2.resize(frame, (width, height))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        inference_time = time.time()
        infer_network.exec_net(0, p_frame)
        if infer_network.wait() == 0:
            observations = []
            outputs = infer_network.get_output()
            total_inference_time.append(time.time() - inference_time)
            print(np.mean(total_inference_time))

            for result in outputs:
                observations = parseYoloV3(result, prob_threshold, cap_h,
                                           cap_w)

            for idx_x, obs_x in enumerate(observations):
                if obs_x.confidence <= 0:
                    continue
                for idx_y, obs_y in enumerate(observations[idx_x + 1:]):
                    intersection = IntersectionOverUnion(obs_x, obs_y)
                    if intersection >= 0.4:
                        observations[idx_y].confidence = 0
            for obs in observations:
                people_in_frame += 1
                label = labels[obs.class_id]
                # Found person
                if (obs.confidence > prob_threshold) & (label == "person"):
                    found_person = True
                    for idx, person in enumerate(found_people):
                        # Check previously found people
                        intersection = IntersectionOverUnion(obs, person)
                        if intersection >= 0.3:
                            obs.time_found = person.time_found
                            obs.last_updated = curr_time
                            found_people[idx] = obs
                            found_person = False
                            break

                    if found_person or len(found_people) == 0:
                        obs.time_found = curr_time
                        obs.last_updated = curr_time
                        found_people.append(obs)
                        people_count += 1

            found_people = [
                person for person in found_people
                if curr_time - person.last_updated < 3
            ]

            # Drawing boxes
            for person in found_people:
                label = person.class_id
                confidence = person.confidence
                if confidence > 0.2:
                    label_text = labels[label] + " (" + "{:.1f}".format(
                        confidence * 100) + "%)"
                    cv2.rectangle(frame, (person.xmin, person.ymin),
                                  (person.xmax, person.ymax), (125, 250, 0), 1)
                    cv2.putText(frame, label_text,
                                (person.xmin, person.ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                1)

            client.publish(
                "person",
                json.dumps({
                    "count": len(found_people),
                    "total": people_count
                }))
            for person in found_people:
                t = curr_time - person.time_found
                client.publish("person/duration", json.dumps({"duration": t}))
            if flag:
                sys.stdout.buffer.write(frame)
                sys.stdout.flush()
            if key_pressed == 27:
                break


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
