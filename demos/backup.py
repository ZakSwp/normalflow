import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import numpy as np
import yaml

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow, LoseTrackError
from normalflow.utils import Frame, render_surface_info_video, intialize_debug_folders
from normalflow.viz_utils import annotate_coordinate_system


import re
import fcntl
import struct

import time
VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001

folderList = ["/home/zakaria/Desktop/normalflow/debug_diff/collapsed", "/home/zakaria/Desktop/normalflow/debug_diff/raw"]
intialize_debug_folders(folderList)
SENSOR_MAP = {
    "DIGIT":        ["/home/zakaria/Desktop/normalflow/demos/configs/digit.yaml", "/home/zakaria/Desktop/normalflow/demos/models/digit/nnmodel_digit_2.pth"],
    "GelSight Mini": ["/home/zakaria/Desktop/normalflow/demos/configs/gsmini.yaml","nnmodel.pth"], #replace model with nnmodel.pth
 }

#"/home/zakaria/Desktop/normalflow/demos/models/gsmini/nnmodel_gsmini_decalib.pth"
def is_capture_node(dev_id):
    fmt = "16s32s32sII4I"
    buf = struct.pack(fmt, b"", b"", b"", 0, 0, 0, 0, 0, 0)
    try:
        with open(f"/dev/video{dev_id}", "rb") as f:
            result = fcntl.ioctl(f, VIDIOC_QUERYCAP, buf)
        caps = struct.unpack(fmt, result)[4]
        return bool(caps & V4L2_CAP_VIDEO_CAPTURE)
    except Exception:
        return False

def detect_sensor():
    for file in sorted(os.listdir("/sys/class/video4linux")):
        name_path = os.path.realpath(f"/sys/class/video4linux/{file}/name")
        with open(name_path) as f:
            sysfs_name = f.read().strip()
        dev_id = int(re.search(r"\d+$", file).group(0))
        if not is_capture_node(dev_id):
            continue
        for sensor_name, file_list in SENSOR_MAP.items():
            if sensor_name in sysfs_name:
                return {
                    "sensor": sensor_name,
                    "device_index": dev_id,
                    "device_path": f"/dev/video{dev_id}",
                    "config": file_list[0],
                    "model": file_list[1],
                }

    return None


"""
Usage:
    python realtime_object_tracking.py [--calib_model_path CALIB_MODEL_PATH] [--config_path CONFIG_PATH] [--device {cpu, cuda}]

Press any key to quit the streaming session.
"""


detected = detect_sensor()
if detected is None:
    raise RuntimeError("No tactile sensor found.")

calib_model_path = os.path.join(os.path.dirname(__file__), "models", detected["model"]) #nnmodel.pth
config_path = os.path.join(os.path.dirname(__file__), "configs", detected["config"]) #gsmini.yaml
print("Running normalflow with {} sensor, {} config file".format(detected["sensor"],detected["config"]))

def resize_show(image, frame_name="frame", scale=2.5):
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(frame_name, image)


def realtime_object_tracking():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Real-time tracking the object using tactile sensors."
    )
    parser.add_argument(
        "-b",
        "--calib_model_path",
        type=str,
        help="place",
        default=calib_model_path,
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path",
        default=config_path,
    )
    parser.add_argument(
        "-s",
        "--streamer",
        type=str,
        choices=["opencv", "ffmpeg"],
        help="The",
        default="opencv",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="the",
        default="cpu",
    )
    args = parser.parse_args()

    # Read the configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        #device_name = input("Type device name (i.e. DIGIT or GelSight Mini)")
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]
        raw_imgh = config["raw_imgh"]
        raw_imgw = config["raw_imgw"]
        framerate = config["framerate"]

    # Connect to the sensor and the reconstructor
    if args.streamer == "opencv":
        device = Camera(device_name, imgh, imgw)
    elif args.streamer == "ffmpeg":
        device = FastCamera(device_name, imgh, imgw, raw_imgh, raw_imgw, framerate)
    device.connect()
    recon = Reconstructor(args.calib_model_path, device="cpu")

    # Collect background images
    print("Collecting 10 background images, please wait ...")
    bg_images = []
    for _ in range(10):
        image = device.get_image()
        bg_images.append(image)
    bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
    recon.load_bg(bg_image)
    print("Done with background collection.")

    # Real-time object tracking
    print("\nStart object tracking, Press any key to quit.\n")
    is_running = True
    frames = []
    try:
        while is_running:
            image = device.get_image()
            G, H, C = recon.get_surface_info(image, ppmm)
            frames.append((G, H, C))
            #print("No contact. {} > {}".format(np.sum(C), 500)) #TODO: Delete

            print("RECORDING")
            frame = Frame(G, H, C)
            if not frame.is_contacted:
                resize_show(image)
                key = cv2.waitKey(1)
                if key != -1:
                    is_running = False
                continue
            else:
                # Tracking a new object, wait 2 frames for the contact to stabilize
                for _ in range(2):
                    image = device.get_image()
                    resize_show(image)
                    key = cv2.waitKey(1)

                # Get the surface information of the reference frame (key frame)
                image_start = device.get_image()
                G_start, H_start, C_start = recon.get_surface_info(image_start, ppmm)
                frames.append((G_start, H_start, C_start))
                print("RECORDING")

                frame_start = Frame(G_start, H_start, C_start)
                # For display purpose, get the largest contour and its center
                contours_start, _ = cv2.findContours(
                    (C_start * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                M_start = cv2.moments(max(contours_start, key=cv2.contourArea))
                cx_start, cy_start = int(M_start["m10"] / M_start["m00"]), int(
                    M_start["m01"] / M_start["m00"]
                )

                # Start tracking this object relative to the reference frame (key frame)
                frame_ref = frame_start
                frame_prev = frame_start
                prev_T_ref = np.eye(4, dtype=np.float32)
                start_T_ref = np.eye(4, dtype=np.float32)
                is_tracking = True
                print("STARTING VALUES: C Sum: {}".format(np.sum(C_start)))
                while is_tracking:
                    # Get the surface information of the current frame
                    image_curr = device.get_image()
                    G_curr, H_curr, C_curr = recon.get_surface_info(image_curr, ppmm)
                    frames.append((G_curr, H_curr, C_curr))
                    print("RECORDING")
                    frame_curr = Frame(G_curr, H_curr, C_curr)
                    if not frame_curr.is_contacted:
                        is_tracking = False
                        print("STOPPED TRACKING")
                        break
                    print("Frame contact verified. {} > {}".format(np.sum(frame_curr.C), 500)) #TODO: Delete

                    # Use NormalFlow to estimate the transformation
                    try:
                        curr_T_ref = normalflow(
                            frame_ref.N,
                            frame_ref.C,
                            frame_ref.H,
                            frame_ref.L,
                            frame_curr.N,
                            frame_curr.C,
                            frame_curr.H,
                            frame_curr.L,
                            prev_T_ref,
                            ppmm,
                            #scr_threshold=1.00,
                            #ccs_threshold=1.00,
                        )
                        #print("NORMAL FLOW WITH HIGH THRESHOLD FOR TRANSFORMATION ESTIMATION") #TODO: Delete
                        frame_prev = frame_curr
                        prev_T_ref = curr_T_ref
                    except LoseTrackError:
                        # Reset reference frame as the previous frame
                        #print("SETTING REF FRAME AS PREV") #TODO: Delete
                        frame_ref = frame_prev
                        start_T_ref = start_T_ref @ np.linalg.inv(prev_T_ref)
                        prev_T_ref = np.eye(4, dtype=np.float32)
                        # Use NormalFlow to estimate the transformation to the new reference frame
                        try:
                            # We disable the threshold for consecutive frame tracking
                            curr_T_ref = normalflow(
                                frame_ref.N,
                                frame_ref.C,
                                frame_ref.H,
                                frame_ref.L,
                                frame_curr.N,
                                frame_curr.C,
                                frame_curr.H,
                                frame_curr.L,
                                prev_T_ref,
                                ppmm,
                                scr_threshold=0.0,
                                ccs_threshold=0.0,
                            )
                            print("DISABLING THRESHOLD FOR CONSEC FRAME TRACKING") #TODO: Delete
                            frame_prev = frame_curr
                            prev_T_ref = curr_T_ref
                        except LoseTrackError:
                            # Lose track, set current frame as new start frame
                            print("Lose Track!")
                            is_tracking = False
                            break
                    
                    #time.sleep(0.1) #TODO: Delete
                    # Display the object tracking result
                    image_l = image_start.copy()
                    cv2.putText(
                        image_l,
                        "Initial Frame",
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    center_start = np.array([cx_start, cy_start]).astype(np.int32)
                    unit_vectors_start = np.eye(3)[:, :2]
                    annotate_coordinate_system(image_l, center_start, unit_vectors_start)
                    # Annotate the transformation on the target frame
                    image_r = image_curr.copy()
                    cv2.putText(
                        image_r,
                        "Current Frame",
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    center_3d_start = (
                        np.array(
                            [(cx_start - imgw / 2 + 0.5), (cy_start - imgh / 2 + 0.5), 0]
                        )
                        * ppmm
                        / 1000.0
                    )
                    unit_vectors_3d_start = np.eye(3) * ppmm / 1000.0
                    curr_T_start = curr_T_ref @ np.linalg.inv(start_T_ref)
                    remapped_center_3d_start = (
                        np.dot(curr_T_start[:3, :3], center_3d_start) + curr_T_start[:3, 3]
                    )
                    remapped_cx_start = (
                        remapped_center_3d_start[0] * 1000 / ppmm + imgw / 2 - 0.5
                    )
                    remapped_cy_start = (   
                        remapped_center_3d_start[1] * 1000 / ppmm + imgh / 2 - 0.5
                    )
                    remapped_center_start = np.array(
                        [remapped_cx_start, remapped_cy_start]
                    ).astype(np.int32)
                    remapped_unit_vectors_start = (
                        np.dot(curr_T_start[:3, :3], unit_vectors_3d_start.T).T
                        * 1000
                        / ppmm
                    )[:, :2]
                    annotate_coordinate_system(
                        image_r, remapped_center_start, remapped_unit_vectors_start
                    )

                    # Display
                    resize_show(cv2.hconcat([image_l, image_r]))
                    key = cv2.waitKey(1)
                    if key != -1:
                        is_tracking = False
                        is_running = False
    except Exception as e:
        print("Reconstruction error: {}".format(e))
        try:
            # --- DEBUG ---
            debug_canvas = np.zeros_like((C * 255).astype(np.uint8))
            cv2.drawContours(debug_canvas, contours_start, -1, 255, 1)
            cv2.imshow("C mask", (C * 255).astype(np.uint8))
            cv2.imshow("findContours result", debug_canvas)
            cv2.waitKey(0)  # blocks until keypress — change to cv2.waitKey(1) if you want non-blocking
            print(f"Number of contours: {len(contours_start)}")
            for i, c in enumerate(contours_start):
                print(f"  contour {i}: {len(c)} points, area={cv2.contourArea(c):.2f}")
            # --- END DEBUG ---
        except:
            pass
        pass
    device.release()
    cv2.destroyAllWindows()
    render_surface_info_video(frames, output_path='tracking_run.mp4', fps=10)


if __name__ == "__main__":
    realtime_object_tracking()
