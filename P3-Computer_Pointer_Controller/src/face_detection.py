'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from pprint import PrettyPrinter
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin, IECore


class Model_FD:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device, extensions):

        self.plugin = None
        self.net = None
        self.input_blob = None
        self.output_blob = None
        self.exec_net = None
        self.infer_request = None
        self.input_shape = None
        self.output_shape = None
        self.input_name = None
        self.device = device
        self.extension = extensions
        self.model = model_name

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model + ".xml"
        model_weights = self.model + ".bin"

        self.plugin = IECore()
        self.net = IENetwork(model_xml, model_weights)

        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device,
                                                 num_requests=1)

        if self.extension and 'CPU' in self.device:
            self.plugin.add_cpu_extension(self.extension)
        self.check_model()

        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))

        self.input_shape = self.net.inputs[self.input_blob].shape
        self.output_shape = self.net.outputs[self.output_blob].shape

    def predict(self, image, benchmark_timing):
        '''
        This method is meant for running predictions on the input image.
        '''
        global outputs
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: image})

        if self.exec_net.requests[0].wait(-1) == 0:
            outputs = self.exec_net.requests[0].outputs[self.output_blob]

            if benchmark_timing:
                pp = PrettyPrinter(indent=4)
                print('Benchmark Timing for Face_Detection')
                pp.pprint(self.exec_net.requests[0].get_perf_counts())
                # Write get_perf_counts() data to a text file
                data = (self.exec_net.requests[0].get_perf_counts())
                self.write_benchmark('Benchmark Timing for Face_Detection', data)

        return outputs

    def check_model(self):

        supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        global pimage
        try:
            pimage = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)
            pimage = pimage.transpose((2, 0, 1))
            pimage = pimage.reshape(image, (1, *image.shape))
        except Exception as e:
            log.error(str(e))
        return pimage

    def preprocess_output(self, image, outputs, threshold, display):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        global h, w
        try:
            h = image.shape[0]
            w = image.shape[1]
        except Exception as e:
            log.error(str(e))

        facebox = []
        # Drawing the box/boxes
        for obj in outputs[0][0]:
            if obj[2] > threshold:

                if obj[3] < 0:
                    obj[3] = -obj[3]
                if obj[4] < 0:
                    obj[4] = -obj[4]

                xmin = int(obj[3] * w)
                ymin = int(obj[4] * h)
                xmax = int(obj[5] * w)
                ymax = int(obj[6] * h)

                # Drawing the box in the image
                if display:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                facebox.append([xmin, ymin, xmax, ymax])

        return facebox

    def get_model_name(self):
        return self.model

    def write_benchmark(self, title, data):

        with open("FaceDetectione_benchmark_timing.txt", "a") as f:
            f.write(str(title) + "\n")
            f.write(str(data) + '\n')
            f.close()
