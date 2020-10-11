'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import math

import cv2
import numpy as np
import logging as log
import time
from openvino.inference_engine import IENetwork, IEPlugin, IECore


class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        #self.model_xml = model_name
        self.device = device
        self.extensions = extensions
        # Initialise the class
        self.infer_network = None
        # raise NotImplementedError
        self.model = model_name
        self.device = device
        self.extension = extensions


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model + ".xml"
        model_weights = self.model + ".bin"
        self.plugin = IECore()
        self.net = IENetwork(model_xml, model_weights)

        #self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)

        try:
            self.core = IECore()
            self.exec_net = self.core.load_network(network=self.net, device_name=self.device, num_requests=1)
            if self.extension is not None:
                self.core.add_extension(extension_path=self.extension, device_name=self.device)
        except Exception as e:
            print('Error occurred, refer `CPC.log` file for details')
            log.error('Head Pose Estimation IECore object could not be initialized/loaded.', e)
        return
        # raise NotImplementedError

    def predict(self, left_eye_image, right_eye_image, headpose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        input_dict = {'left_eye_image': left_eye_image,
                      'right_eye_image': right_eye_image,
                      'head_pose_angles': headpose_angles}

        self.infer_network= self.exec_net.start_async(request_id=0, inputs=input_dict)

        # Wait for the result
        if self.infer_network.wait() == 0:
            # end time of inference
            self.result = self.exec_net.requests[0].outputs
            return self.result

    def check_model(self):
        supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error('Unsupported layers found:{}'.format(unsupported_layers))
            log.error('Check for any extensions for these unsupported layers available for adding to IECore')
            exit(1)

    def preprocess_input1(self,  face, left_eye_point, right_eye_point):
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

        lefteye_input_shape = [1, 3, 60, 60]  # self.infer_network.get_input_shape()
        righteye_input_shape = [1, 3, 60, 60]  # self.infer_network.get_next_input_shape(2)

        # crop left eye
        l_x_center = left_eye_point[0]
        l_y_center = left_eye_point[1]
        width = lefteye_input_shape[3]
        height = lefteye_input_shape[2]
        # ymin:ymax, xmin:xmax
        print('faceedge ', face[0][0])
        facewidthedge = face.shape[1]
        faceheightedge = face.shape[0]
        print('facewidthedge ',facewidthedge)
        print('faceheightedge ',faceheightedge)



        # check for edges to not crop
        ymin = int(l_y_center - height // 2) if int(l_y_center - height // 2) >= 0 else 0
        ymax = int(l_y_center + height // 2) if int(l_y_center + height // 2) <= faceheightedge else faceheightedge

        xmin = int(l_x_center - width // 2) if int(l_x_center - width // 2) >= 0 else 0
        xmax = int(l_x_center + width // 2) if int(l_x_center + width // 2) <= facewidthedge else facewidthedge

        left_eye_image = face[ymin: ymax, xmin:xmax]
        print('left_eye_image ',left_eye_image[0][0])
        # print out left eye to frame

        # left eye [1x3x60x60]
        p_frame_left = cv2.resize(left_eye_image[0][0], (lefteye_input_shape[3], lefteye_input_shape[2]))
        print('pframeleft ', p_frame_left)
        p_frame_left = p_frame_left.transpose((2, 0, 1))
        p_frame_left = p_frame_left.reshape(1, *p_frame_left.shape)

        # crop right eye
        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        width = righteye_input_shape[3]
        height = righteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        # check for edges to not crop
        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= faceheightedge else faceheightedge

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= facewidthedge else facewidthedge

        right_eye_image = face[ymin: ymax, xmin:xmax]
        # print out left eye to frame

        # right eye [1x3x60x60]
        p_frame_right = cv2.resize(right_eye_image, (righteye_input_shape[3], righteye_input_shape[2]))
        p_frame_right = p_frame_right.transpose((2, 0, 1))
        p_frame_right = p_frame_right.reshape(1, *p_frame_right.shape)

        # headpose_angles

        return  p_frame_left, p_frame_right
        # raise NotImplementedError

    def preprocess_input(self, left_eye_image, right_eye_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        print('left_eye_image ',left_eye_image)

        left_eye_pre_image = cv2.resize(left_eye_image, (60, 60))
        left_eye_pre_image = left_eye_pre_image.transpose((2, 0, 1))
        left_eye_pre_image = left_eye_pre_image.reshape(1, *left_eye_pre_image.shape)

        right_eye_pre_image = cv2.resize(right_eye_image, (60, 60))
        right_eye_pre_image = right_eye_pre_image.transpose((2, 0, 1))
        right_eye_pre_image = right_eye_pre_image.reshape(1, *right_eye_pre_image.shape)

        return left_eye_pre_image, right_eye_pre_image

    def preprocess_input3(self, frame, face, left_eye_point, right_eye_point, print_flag=True):
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

        lefteye_input_shape = [1, 3, 60, 60]  # self.infer_network.get_input_shape()
        righteye_input_shape = [1, 3, 60, 60]  # self.infer_network.get_next_input_shape(2)

        # crop left eye
        x_center = left_eye_point[0]
        y_center = left_eye_point[1]
        width = lefteye_input_shape[3]
        height = lefteye_input_shape[2]
        # ymin:ymax, xmin:xmax
        facewidthedge = face[0].shape[1]
        faceheightedge = face[0].shape[0]

        # check for edges to not crop
        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= faceheightedge else faceheightedge

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= facewidthedge else facewidthedge

        left_eye_image = face[ymin: ymax, xmin:xmax]
        # print out left eye to frame

        #left_eye_image=frame[150:150 + left_eye_image.shape[0], 20:20 + left_eye_image.shape[1]]
        # left eye [1x3x60x60]
        p_frame_left = cv2.resize(left_eye_image, (lefteye_input_shape[3], lefteye_input_shape[2]))
        p_frame_left = p_frame_left.transpose((2, 0, 1))
        p_frame_left = p_frame_left.reshape(1, *p_frame_left.shape)

        # crop right eye
        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        width = righteye_input_shape[3]
        height = righteye_input_shape[2]
        # ymin:ymax, xmin:xmax
        # check for edges to not crop
        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= faceheightedge else faceheightedge

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= facewidthedge else facewidthedge

        right_eye_image = face[ymin: ymax, xmin:xmax]
        # print out left eye to frame

        #right_eye_image=frame[150:150 + right_eye_image.shape[0], 100:100 + right_eye_image.shape[1]]

        # right eye [1x3x60x60]
        p_frame_right = cv2.resize(right_eye_image, (righteye_input_shape[3], righteye_input_shape[2]))
        p_frame_right = p_frame_right.transpose((2, 0, 1))
        p_frame_right = p_frame_right.reshape(1, *p_frame_right.shape)

        # headpose_angles

        return frame, p_frame_left, p_frame_right


    def preprocess_output(self, outputs, image, facebox, left_eye_point, right_eye_point):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.
        Output layer name in Inference Engine format:
        gaze_vector
        '''
        gaz_result= outputs['gaze_vector'][0]
        x = gaz_result[0]
        y = gaz_result[1]
        z = gaz_result[2]
        # Draw output

        cv2.putText(image,
                    "x:" + str('{:.1f}'.format(x * 100)) + ",y:" + str('{:.1f}'.format(y * 100)) + ",z:" + str(
                        '{:.1f}'.format(z)), (20, 100), 0, 0.6, (0, 0, 255), 1)

        # left eye
        xmin, ymin, _, _ = facebox
        x_center = left_eye_point[0]
        y_center = left_eye_point[1]
        left_eye_center_x = int(xmin + x_center)
        left_eye_center_y = int(ymin + y_center)

        # right eye
        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        right_eye_center_x = int(xmin + x_center)
        right_eye_center_y = int(ymin + y_center)

        cv2.arrowedLine(image, (left_eye_center_x, left_eye_center_y),
                            (left_eye_center_x + int(x * 100), left_eye_center_y + int(-y * 100)), (255, 100, 100), 5)
        cv2.arrowedLine(image, (right_eye_center_x, right_eye_center_y),
                            (right_eye_center_x + int(x * 100), right_eye_center_y + int(-y * 100)), (255, 100, 100), 5)

        return [x, y, z]

    def preprocess_input5(self, image):

        p_frame = cv2.resize((image), 60, 60)
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output5(self, outputs, head_position):

        roll = head_position[2]
        gaze_vector = outputs

        cos_theta = math.cos(roll * math.pi / 180)
        sin_theta = math.sin(roll * math.pi / 180)

        x = outputs[0] * cos_theta + outputs[1] * sin_theta
        y = outputs[1] * cos_theta - outputs[0] * sin_theta

        return (x, y), gaze_vector