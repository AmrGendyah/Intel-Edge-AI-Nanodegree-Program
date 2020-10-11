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
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device, num_requests, cpu_extension=None, plugin=None):
        ### Load the Inference Engine API
        model_xml = model

        ### Load the model ###
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device=device)
        else:
            self.plugin = plugin
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

        ### Check for supported layers ###
        supported_layers = self.plugin.get_supported_layers(self.network)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            exit(1)

        print("IR successfully loaded into Inference Engine.")

        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.exec_network = self.plugin.load(network=self.network)
        else:
            self.exec_network = self.plugin.load(network=self.network, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.get_input_shape()

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id, image):

        self.infer_request_handle = self.exec_network.start_async(
            request_id=request_id, inputs={self.input_blob: image})
        return self.exec_network

    def wait(self):

        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):

        return self.exec_network.requests[0].outputs[self.output_blob]

    def performance_counter(self):

        perf_count = self.exec_network.requests[0].get_perf_counts()
        return perf_count

    def reset(self):

        del self.exec_network
        del self.plugin
        del self.network