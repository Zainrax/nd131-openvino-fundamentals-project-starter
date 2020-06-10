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
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

anchors = [
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373,
    326
]
classes = 80


class DetectionObservation():
    time_found = 0.0
    last_updated = 0.0

    def __init__(self, xmin, ymin, xmax, ymax, conf, label):
        self.confidence = conf
        self.class_id = label
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    def get_area(self):
        area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        return area

    def calc_iou(self, rect):
        min_X = min(self.xmax, rect.xmax)
        max_X = max(self.xmin, rect.xmin)
        min_Y = min(self.ymax, rect.ymax)
        max_Y = max(self.ymin, rect.ymin)

        area_of_intersection = abs((max_X - min_X) * (max_Y - min_Y))
        if area_of_intersection == 0:
            return 0
        iou = area_of_intersection / float(rect.get_area() + self.get_area() -
                                           area_of_intersection)
        return iou


def parseResult(result, threshold, w_scale, h_scale):
    predictions = result[0][0]
    observations = []
    for p in predictions:
        conf = p[2]
        if conf > threshold:
            print(conf)
            label = p[1]
            xmin = p[3] * w_scale
            ymin = p[4] * h_scale
            xmax = p[5] * w_scale
            ymax = p[6] * h_scale
            obs = DetectionObservation(xmin, ymin, xmax, ymax, conf, label)
            observations.append(obs)
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
    infer_network.load_model(args.model, args.cpu_extension, args.device)

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get frame inforamtion
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_shape = infer_network.get_input_shape()
    num = input_shape[0]
    channel = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

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
        p_frame = p_frame.reshape(num, channel, height, width)
        cv2.cvtColor(p_frame, cv2.COLOR_RGB2BGR)

        inference_time = time.time()
        infer_network.exec_net(0, p_frame)
        if infer_network.wait() == 0:
            observations = []
            outputs = infer_network.get_output()
            total_inference_time.append(time.time() - inference_time)
            # print(np.mean(total_inference_time))
            # print(curr_time)

            for result in outputs:
                observations = parseResult(result, prob_threshold, cap_w,
                                           cap_h)

            for idx_x, obs_x in enumerate(observations):
                if obs_x.confidence <= 0:
                    continue
                for idx_y, obs_y in enumerate(observations[idx_x + 1:]):
                    intersection = obs_x.calc_iou(obs_y)
                    if intersection >= 0.2:
                        observations[idx_y].confidence = 0

            for obs in observations:
                people_in_frame += 1
                # class_id at 0 is Person
                if (obs.confidence > prob_threshold):
                    found_person = True
                    for idx, person in enumerate(found_people):
                        # Check previously found people
                        intersection = obs.calc_iou(person)
                        print(intersection)
                        if intersection >= 0.3:
                            # Found previous person, update position
                            obs.time_found = person.time_found
                            obs.last_updated = curr_time
                            found_people[idx] = obs
                            found_person = False
                            break

                    # Add found person to list of activty observations
                    if found_person or len(found_people) == 0:
                        obs.time_found = curr_time
                        obs.last_updated = curr_time
                        found_people.append(obs)
                        people_count += 1

            # Filter out people that have not been in the frame for 3 seconds
            left_people = [
                person for person in found_people
                if curr_time - person.last_updated >= 3
            ]
            found_people = [
                person for person in found_people
                if curr_time - person.last_updated < 3
            ]
            print(len(found_people))
            client.publish("person", json.dumps({"count": 1}))
            # Draw boxes
            for person in found_people:
                cv2.rectangle(frame, (person.xmin, person.ymin),
                              (person.xmax, person.ymax), (125, 250, 0), 1)

            for person in left_people:
                t = curr_time - person.time_found
                client.publish("person/duration", json.dumps({"duration": t}))
        frame = cv2.resize(frame, (cap_w, cap_h))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        if key_pressed == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


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
