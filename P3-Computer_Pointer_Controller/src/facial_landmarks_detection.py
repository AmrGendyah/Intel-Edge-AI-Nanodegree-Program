'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from pprint import PrettyPrinter
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin, IECore


class Model_FLD:
    '''
    Class for the Face Landmarks Detection Model.
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
        self.device = device
        self.extension = extensions
        self.model = model_name
        self.outputs = None
        self.eyes_coords_dict = None

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

        self.exec_net.start_async(request_id=0, inputs={self.input_blob: image})
        if self.exec_net.requests[0].wait(-1) == 0:
            self.outputs = self.exec_net.requests[0].outputs[self.output_blob]

            if benchmark_timing:
                pp = PrettyPrinter(indent=4)
                print('Benchmark Timing for Facial_Landmark_Detection')
                pp.pprint(self.exec_net.requests[0].get_perf_counts())
                # Write get_perf_counts() data to a text file
                data = (self.exec_net.requests[0].get_perf_counts())
                self.write_benchmark('Benchmark Timing for Facial_Landmark_Detection', data)

        return self.outputs

    def check_model(self):
        supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error('Unsupported layers found:{}'.format(unsupported_layers))
            log.error('Check for any extensions for these unsupported layers available for adding to IECore')
            exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        global pimage
        try:
            pimage = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            pimage = pimage.transpose((2, 0, 1))
            pimage = pimage.reshape(pimage, (1, *pimage.shape))
        except Exception as e:
            log.error(str(e))

        return pimage

    def preprocess_output(self, image, outputs, facebox, display):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values
        for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
         All the coordinates are normalized to be in range [0,1]
        '''
        """
        # Output layer names in Inference Engine format:
        # landmarks-regression-retail-0009:
        #   "95", [1, 10, 1, 1], containing a row-vector of 10 floating point values for five landmarks
        #         coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        #         All the coordinates are normalized to be in range [0,1]
        """

        normed_landmarks = outputs.reshape(1, 10)[0]

        height = facebox[3] - facebox[1]
        width = facebox[2] - facebox[0]

        if display:
            # Drawing the box/boxes
            for i in range(2):
                x = int(normed_landmarks[i * 2] * width)
                y = int(normed_landmarks[i * 2 + 1] * height)
                cv2.circle(image, (facebox[0] + x, facebox[1] + y), 30, (0, 255, i * 255), 1)

        left_eye_point = [normed_landmarks[0] * width, normed_landmarks[1] * height]
        right_eye_point = [normed_landmarks[2] * width, normed_landmarks[3] * height]

        return left_eye_point, right_eye_point

    def get_model_name(self):
        return self.model

    def write_benchmark(self, title, data):

        with open("FacialLandmarks_benchmark_timing.txt", "a") as f:
            f.write(str(title) + "\n")
            f.write(str(data) + '\n')
            f.close()
