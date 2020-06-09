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

anchors = [
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373,
    326
]
classes = 80


class DetectionObservation():
    time_found = 0.0
    last_updated = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, w_scale, h_scale):
        self.confidence = confidence
        self.class_id = class_id
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)

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


def parseYoloV3(out, threshold, cap_h, cap_w):
    observations = []
    grid_side = out.shape[2]

    offset = 0
    if grid_side == 13:
        offset = 2 * 6
    if grid_side == 26:
        offset = 2 * 3
    if grid_side == 52:
        offset = 2 * 0

    grid = out.transpose((0, 2, 3, 1))
    for row_idx, row in enumerate(grid[0]):
        for col_idx, col in enumerate(row):
            # 3 is the number of bounding boxes
            bounding_boxes = np.split(col, 3)
            for idx, box in enumerate(bounding_boxes):
                x = (box[0] + col_idx) / grid_side * 416
                y = (box[1] + row_idx) / grid_side * 416
                w = math.exp(box[2]) * anchors[offset + 2 * idx]
                h = math.exp(box[3]) * anchors[offset + 2 * idx + 1]
                p = box[4]
                if p < threshold:
                    continue
                class_id = np.argmax(box[5:])
                observation = DetectionObservation(x, y, h, w, class_id, p,
                                                   (cap_h / 416),
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
            # print(np.mean(total_inference_time))

            for result in outputs:
                observations = parseYoloV3(result, prob_threshold, cap_h,
                                           cap_w)

            for idx_x, obs_x in enumerate(observations):
                if obs_x.confidence <= 0:
                    continue
                for idx_y, obs_y in enumerate(observations[idx_x + 1:]):
                    intersection = obs_x.calc_iou(obs_y)
                    if intersection >= 0.4:
                        observations[idx_y].confidence = 0

            for obs in observations:
                people_in_frame += 1
                # class_id at 0 is Person
                if (obs.confidence > prob_threshold) & (obs.class_id == 0):
                    found_person = True
                    for idx, person in enumerate(found_people):
                        # Check previously found people
                        intersection = obs.calc_iou(person)
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
            found_people = [
                person for person in found_people
                if curr_time - person.last_updated < 3
            ]

            # Draw boxes
            for person in found_people:
                print(len(found_people))
                cv2.rectangle(frame, (person.xmin, person.ymin),
                              (person.xmax, person.ymax), (125, 250, 0), 1)

            client.publish(
                "person",
                json.dumps({
                    "count": len(found_people),
                    "total": people_count
                }))
            for person in found_people:
                t = curr_time - person.time_found
                client.publish("person/duration", json.dumps({"duration": t}))
        frame = cv2.resize(frame, (768, 432))
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
