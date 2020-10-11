#!/usr/bin/env python3

from statistics import mean
import cv2
import time
from src.input_feeder import InputFeeder
from src.face_detection import Model_FD
from src.gaze_estimation import Model_GE
from src.facial_landmarks_detection import Model_FLD
from src.head_pose_estimation import Model_HPE
from src.mouse_controller import MouseController
from argparse import ArgumentParser
import logging as log


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-mfd", "--fdmodel", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-mhp", "--hpmodel", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-mlm", "--lmmodel", required=True, type=str,
                        help="Path to a facial landmarks xml file with a trained model.")
    parser.add_argument("-mge", "--gemodel", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path video file or CAM to use camera")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                             "(0.6 by default)")

    parser.add_argument("-dp", "--display_outputs", type=bool, default=False, required=False,
                        help="Display the outputs of the models.")

    parser.add_argument("-bmfd", "--benchmark_face_detection", type=bool, default=False, required=False,
                        help="Print the time it takes for each layer in face_detection model")

    parser.add_argument("-bmfl", "--benchmark_facial_landmark_detection", type=bool, default=False, required=False,
                        help="Print the time it takes for each layer in  facial_landmark_detection model")

    parser.add_argument("-bmhp", "--benchmark_head_pose_estimation", type=bool, default=False, required=False,
                        help="Print the time it takes for each layer in head_pose_estimation model")

    parser.add_argument("-bmge", "--benchmark_gaze_estimation", type=bool, default=False, required=False,
                        help="Print the time it takes for each layer in gaze_estimation model")

    return parser


def main(args):
    global gazevector, AVfdtime, AVlmtime, AVhptime, AVgetime
    fd_infertime = [0]
    lm_infertime = [0]
    hp_infertime = [0]
    ge_infertime = [0]

    model_fd = args.fdmodel
    model_ge = args.gemodel
    model_hpe = args.hpmodel
    model_fld = args.lmmodel
    input_feeder = args.input
    device = args.device
    CPUextension = args.cpu_extension
    prob_threshold = args.prob_threshold
    display_output = args.display_outputs
    FD_bench = args.benchmark_face_detection
    FLE_bench = args.benchmark_facial_landmark_detection
    HPE_bench = args.benchmark_head_pose_estimation
    GE_bench = args.benchmark_gaze_estimation

    Face_Det = Model_FD(model_fd, device, CPUextension)
    start_time = time.time()
    Face_Det.load_model()
    fd_loadtime = time.time() - start_time

    Head_Pose = Model_HPE(model_hpe, device, CPUextension)
    start_time = time.time()
    Head_Pose.load_model()
    hd_loadtime = time.time() - start_time

    Landmarks_Det = Model_FLD(model_fld, device, CPUextension)
    start_time = time.time()
    Landmarks_Det.load_model()
    ld_loadtime = time.time() - start_time

    Gaze_Det = Model_GE(model_ge, device, CPUextension)
    start_time = time.time()
    Gaze_Det.load_model(device)
    gd_loadtime = time.time() - start_time

    mc = MouseController("medium", "fast")

    # Create a VideoCapture object and read from input file or cam
    if input_feeder.lower() == 'cam':
        feed = InputFeeder(input_type='cam')
    else:
        feed = InputFeeder("video", input_feeder)

    feed.load_data()

    try:

        for frame in feed.next_batch():
            key = cv2.waitKey(10)

            face_ip = Face_Det.preprocess_input(frame)
            start_time = time.time()
            face_op = Face_Det.predict(face_ip, FD_bench)
            fd_infertime.append(time.time() - start_time)
            faces = Face_Det.preprocess_output(frame, face_op, prob_threshold, display_output)

            for face_box in faces:
                face = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                # Get Eyes coords.
                landmark_ip = Landmarks_Det.preprocess_input(face)
                start_time = time.time()
                landmark_op = Landmarks_Det.predict(landmark_ip, FLE_bench)
                lm_infertime.append(time.time() - start_time)
                left_eye_coord, right_eye_coord = Landmarks_Det.preprocess_output(frame, landmark_op, face_box,
                                                                                  display_output)
                # Get head axes
                head_pose_ip = Head_Pose.preprocess_input(face)
                start_time = time.time()
                head_pose_op = Head_Pose.predict(head_pose_ip, HPE_bench)
                hp_infertime.append(time.time() - start_time)
                axes_op = Head_Pose.preprocess_output(frame, head_pose_op, face_box, face, display_output)
                cv2.putText(frame,
                            "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(axes_op[0], axes_op[1],
                                                                                          axes_op[2]), (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                # Get mouse coords
                left_eye, right_eye = Gaze_Det.preprocess_input(face, left_eye_coord, right_eye_coord)
                start_time = time.time()
                geoutput = Gaze_Det.predict(left_eye, right_eye, axes_op, GE_bench)
                ge_infertime.append(time.time() - start_time)
                gazevector = Gaze_Det.preprocess_output(geoutput, frame, face, face_box, left_eye_coord,
                                                        right_eye_coord, display_output)
                cv2.putText(frame,
                            "Gaze Coords: x:{:.2f} | y:{:.2f} | z:{:.2f}".format(gazevector[0], gazevector[1],
                                                                                 gazevector[2]), (5, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                cv2.imshow('Computer Pointer Window', frame)
                mc.move(gazevector[0], gazevector[1])

            # Press ESC on keyboard to  exit
            if key == 27:
                break

        feed.close()

    except Exception as e:
        log.error(str(e))

    # Write load and inference time for each model in a file
    AVfdtime = round(mean(fd_infertime) * 1000, 4)
    AVlmtime = round(mean(lm_infertime) * 1000, 4)
    AVhptime = round(mean(hp_infertime) * 1000, 4)
    AVgetime = round(mean(ge_infertime) * 1000, 4)

    with open("time_of_models.txt", "w") as f:
        f.write("load time\n")
        f.write(Face_Det.get_model_name() + "= " + str(round(fd_loadtime * 1000, 4)) + " mms" + "\n")
        f.write(Landmarks_Det.get_model_name() + "= " + str(round(ld_loadtime * 1000, 4)) + " mms" + "\n")
        f.write(Head_Pose.get_model_name() + "= " + str(round(hd_loadtime * 1000, 4)) + " mms" + "\n")
        f.write(Gaze_Det.get_model_name() + "= " + str(round(gd_loadtime * 1000, 4)) + " mms" + "\n")

        f.write("Inference time\n")
        f.write(Face_Det.get_model_name() + "= " + str(AVfdtime) + " mms" + "\n")
        f.write(Landmarks_Det.get_model_name() + "= " + str(AVlmtime) + " mms" + "\n")
        f.write(Head_Pose.get_model_name() + "= " + str(AVhptime) + " mms" + "\n")
        f.write(Gaze_Det.get_model_name() + "= " + str(AVgetime) + " mms" + "\n")

        f.close()


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)
