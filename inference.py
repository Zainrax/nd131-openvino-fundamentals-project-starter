#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    def __init__(self):
        self.plugin = None
        self.plugin_net = None
        self.network = None
        self.input_blob = None
        self.output_blob = None

    def load_model(self, model_xml, cpu_ext, device):
        print("Loading IR model into Inference Engine...")
        self.plugin = IECore()
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.network = self.plugin.read_network(model=model_xml,
                                                weights=model_bin)
        if cpu_ext is not None:
            self.plugin.add_extension(cpu_ext, device)
        layers_map = self.plugin.query_network(network=self.network,
                                               device_name=device)
        layers = self.network.layers.keys()
        unsupported_layers = [
            layer for layer in layers_map if layer not in layers
        ]
        if len(unsupported_layers) != 0:
            print("Found unsupported layers: {}".format(unsupported_layers))
            print("Please check the extension for availability.")
            exit(1)

        self.plugin_net = self.plugin.load_network(self.network,
                                                   device,
                                                   num_requests=1000)
        ### Note: You may need to update the function parameters. ###
        print("IR model succesfully loaded into Inference Engine.")
        return

    def get_input_shape(self):
        if self.network is not None:
            self.input_blob = next(iter(self.network.inputs))
            input_shape = self.network.inputs[self.input_blob].shape
            return input_shape
        else:
            print("Network has not been defined")
            exit(1)

    def exec_net(self, request_id, frame):
        if self.input_blob is None:
            print("Unable to make request as input not found.")
            exit(1)
        self.infer_request_handle = self.plugin_net.start_async(
            request_id, inputs={self.input_blob: frame})
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        infer_status = self.infer_request_handle.wait()

        ### Note: You may need to update the function parameters. ###
        return infer_status

    def get_output(self):
        result = self.infer_request_handle.outputs.values()
        ### Note: You may need to update the function parameters. ###
        return result
