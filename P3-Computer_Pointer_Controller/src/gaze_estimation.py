'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from pprint import PrettyPrinter
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin, IECore


class Model_GE:
    '''
    Class for the gaze estimation Model.
    '''

    def __init__(self, model_name, device, extensions):
        self.plugin = None
        self.net = None
        self.exec_net = None
        self.infer_request = None
        self.output_shape = None
        self.device = device
        self.extension = extensions
        self.outputs = None
        self.head_pose_angles = None
        self.left_eye_image = None
        self.right_eye_image = None
        self.model = model_name
        self.head_pose_angles_shape = None
        self.left_eye_shape = None
        self.right_eye_shape = None

    def load_model(self,device):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model + ".xml"
        model_weights = self.model + ".bin"

        self.plugin = IECore()
        self.net = IENetwork(model_xml, model_weights)

        self.exec_net = self.plugin.load_network(network=self.net, device_name=device, num_requests=1)

        if self.extension and 'CPU' in self.device:
            self.plugin.add_cpu_extension(self.extension)
        self.check_model()

        self.input_blob = next(iter(self.net.inputs))
        self.input_blob2 = next(iter(self.net.inputs))

        self.input_shape = self.net.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_blob].shape

    def predict(self, processed_left_eye, processed_right_eye, head_position, benchmark_timing):

        input_dict = {'left_eye_image': processed_left_eye,
                      'right_eye_image': processed_right_eye,
                      'head_pose_angles': head_position}

        infer_request_handle = self.exec_net.start_async(request_id=0, inputs=input_dict)

        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            self.outputs = self.exec_net.requests[0].outputs[self.output_blob][0]

            if benchmark_timing:
                pp = PrettyPrinter(indent=4)
                print('Benchmark Timing for Gaze_detection')
                pp.pprint(self.exec_net.requests[0].get_perf_counts())
                # Write get_perf_counts() data to a text file
                data = pp.pprint(self.exec_net.requests[0].get_perf_counts())
                self.write_benchmark('Benchmark Timing for Gaze_Estimation', data)

        return self.outputs

    def check_model(self):
        supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error('Unsupported layers found:{}'.format(unsupported_layers))
            log.error('Check for any extensions for these unsupported layers available for adding to IECore')
            exit(1)

    def eye_corp(self, face, eyepoint):

        Eye_shape = [1, 3, 60, 60]

        # cropping the eye
        x_center = eyepoint[0]
        y_center = eyepoint[1]
        width = Eye_shape[3]
        height = Eye_shape[2]

        # ymin:ymax, xmin:xmax
        face_width = face.shape[0]
        face_height = face.shape[1]

        face_array = np.array(face)

        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= face_height else face_height

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= face_width else face_width

        eye_image = face_array[ymin:ymax, xmin:xmax]

        return eye_image

    def resize_frame(self, frame):

        global p_frame
        try:

            p_frame = cv2.resize(frame, (60, 60))
            p_frame = p_frame.transpose((2, 0, 1))
            p_frame = p_frame.reshape(1, *p_frame.shape)

        except Exception as e:
            log.error(str(e))

        return p_frame

    def preprocess_input(self, face, left_eye_point, right_eye_point):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
       Blob in the format [BxCxHxW] where:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name left_eye_image and the shape [1x3x60x60].
        Blob in the format [BxCxHxW] where:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name right_eye_image and the shape [1x3x60x60].
        Blob in the format [BxC] where:
        B - batch size
        C - number of channels
        with the name head_pose_angles and the shape [1x3].
        '''

        left_eye_image = self.eye_corp(face, left_eye_point)
        right_eye_image = self.eye_corp(face, right_eye_point)

        # eye shape [1x3x60x60]
        p_frame_left = self.resize_frame(left_eye_image)
        p_frame_right = self.resize_frame(right_eye_image)

        return p_frame_left, p_frame_right

    def preprocess_output(self, outputs, image, face, facebox, left_eye_point, right_eye_point, display):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = np.squeeze(outputs)
        x = output[0]
        y = output[1]
        z = output[2]

        # left eye
        xmin, ymin, _, _ = facebox
        l_x_center = left_eye_point[0]
        l_y_center = left_eye_point[1]
        left_eye_center_x = int(xmin + l_x_center)
        left_eye_center_y = int(ymin + l_y_center)

        # right eye
        r_x_center = right_eye_point[0]
        r_y_center = right_eye_point[1]
        right_eye_center_x = int(xmin + r_x_center)
        right_eye_center_y = int(ymin + r_y_center)

        if display:
            cv2.arrowedLine(image, (left_eye_center_x, left_eye_center_y),
                            (left_eye_center_x + int(x * 100), left_eye_center_y + int(-y * 100)),
                            (255, 0, 0), 3)
            cv2.arrowedLine(image, (right_eye_center_x, right_eye_center_y),
                            (right_eye_center_x + int(x * 100), right_eye_center_y + int(-y * 100)),
                            (0, 255, 0), 3)

        return [x, y, z]

    def get_model_name(self):
        return self.model

    def write_benchmark(self, title, data):

        with open("Gaze_benchmark_timing.txt", "a") as f:
            f.write(str(title) + "\n")
            f.write(str(data) + '\n')
            f.close()
