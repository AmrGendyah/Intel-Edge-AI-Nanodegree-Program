'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from pprint import PrettyPrinter
import cv2
import numpy as np
import logging as log
import math
from openvino.inference_engine import IENetwork, IEPlugin, IECore


class Model_HPE:
    '''
    Class for the Head Pose Estimation Model.
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
        self.output = None

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
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_blob].shape

    def predict(self, image, benchmark_timing):
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: image})
        if self.exec_net.requests[0].wait(-1) == 0:
            self.result = self.exec_net.requests[0].outputs

            if benchmark_timing:
                pp = PrettyPrinter(indent=4)
                print('Benchmark Timing for Head_Pose_Estimation')
                pp.pprint(self.exec_net.requests[0].get_perf_counts())
                # Write get_perf_counts() data to a text file
                data = (self.exec_net.requests[0].get_perf_counts())
                self.write_benchmark('Benchmark Timing for Head_Pose_Estimation', data)

        return self.result

    def check_model(self):
        supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):

        temp = image.copy()
        temp = cv2.resize(temp, (self.input_shape[3], self.input_shape[2]))  # n,c,h,w
        temp = temp.transpose((2, 0, 1))
        temp = temp.reshape(1, *temp.shape)
        return temp

    def preprocess_output(self, image, outputs, facebox, face, display):

        output = []
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])

        pitch = np.squeeze(outputs['angle_p_fc'])
        roll = np.squeeze(outputs['angle_r_fc'])
        yaw = np.squeeze(outputs['angle_y_fc'])
        axes_op = np.array([pitch, roll, yaw])

        if display:
            xmin, ymin, _, _ = facebox
            face_center = (xmin + face.shape[1] / 2, ymin + face.shape[0] / 2, 0)
            self.draw_axes(image, face_center, yaw, pitch, roll)

        return axes_op

    # code source: https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll):
        focal_length = 950.0
        scale = 100

        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                       [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0],
                       [0, 0, 1]])
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        R = Rz @ Ry @ Rx
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame

    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix

    def get_model_name(self):
        return self.model

    def write_benchmark(self, title, data):

        with open("headpose_benchmark_timing.txt", "a") as f:
            f.write(str(title) + "\n")
            f.write(str(data) + '\n')
            f.close()
