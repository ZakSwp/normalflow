import argparse
import os
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import numpy as np
import yaml

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow, LoseTrackError
from normalflow.utils import Frame, render_surface_info_video, intialize_debug_folder, render_subtracted_video, render_video
from normalflow.viz_utils import annotate_coordinate_system


import re
import fcntl
import struct

import time
VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
CLIP_OUTPUT,IDX = intialize_debug_folder("/home/zakaria/Desktop/normalflow/captures/07.14.26/cap_0")
SENSOR_MAP = {
    "DIGIT":        ["./configs/digit.yaml", "./models/digit/nnmodel_digit_2.pth"],
    "GelSight Mini": ["./configs/gsmini.yaml","./models/gsmini/nnmodel.pth"],
 }

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
    File paths default to local config and model paths for gelsight mini and DIGIT.
Press 'r' to start/stop recording (each start/stop cycle is saved as its own clip). Press 'q' to quit the streaming session.
"""

print("Starting modified real-time object tracking demo with contact frame delta and gx/gy gradient recording.")
detected = detect_sensor()
if detected is None:
    raise RuntimeError("No tactile sensor found.")
elif detected["sensor"] in SENSOR_MAP.keys():
    calib_model_path = os.path.join(os.path.dirname(__file__), detected["model"]) #nnmodel.pth
    config_path = os.path.join(os.path.dirname(__file__), detected["config"]) #gsmini.yaml
    print("Running with config file at: {}".format(config_path))    

def resize_show(image, frame_name="frame", scale=2.5, is_recording=False):
    image = image.copy()
    if is_recording:
        cv2.circle(image, (image.shape[1] - 15, 15), 8, (0, 0, 255), -1)
        cv2.putText(
            image,
            "REC",
            (image.shape[1] - 55, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.imshow(frame_name, image)


def show_frame_count(is_recording, count):
    """
    Print the current frame count in place (overwriting the same terminal
    line via a carriage return) so it updates live without spamming the
    terminal with a new line per frame.
    """
    if is_recording:
        text = "Recording... frames captured: {:6d}".format(count)
    else:
        text = "Idle (press 'r' to start recording)"
    # Pad with trailing spaces so a shorter message fully overwrites a longer one.
    sys.stdout.write("\r" + text.ljust(50))
    sys.stdout.flush()


def handle_key(key, is_recording):
    """
    Interpret a key press.
    'r' toggles recording on/off.
    'q' (or ESC) requests the program to quit.
    Returns (is_recording, should_quit, just_stopped), where just_stopped is
    True only on the exact frame that recording transitions from on -> off.
    """
    should_quit = False
    just_stopped = False
    if key == -1:
        return is_recording, should_quit, just_stopped
    key &= 0xFF
    if key in (ord("r"), ord("R")):
        was_recording = is_recording
        is_recording = not is_recording
        if was_recording and not is_recording:
            just_stopped = True
        print("\n" + ("Recording STARTED." if is_recording else "Recording STOPPED."))
    elif key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
        should_quit = True
    return is_recording, should_quit, just_stopped


def save_clip(frames, raw_frames, bg_image, clip_index):
    """
    Render and save one recording segment ("clip") to disk immediately.
    Each clip gets its own set of output files so consecutive start/stop
    cycles never get merged into the same video.
    """
    if not frames:
        return
    tag = "clip{:03d}".format(clip_index)
    print("Saving {} ({} frames) ...".format(tag, len(frames)))
    render_surface_info_video(
        frames, output_path=CLIP_OUTPUT + "/{}_{}.mp4".format(IDX,tag), fps=10
    )
    render_subtracted_video(
        raw_frames,
        bg_image,
        output_path=CLIP_OUTPUT + "/{}_{}_sub.mp4".format(IDX,tag),
        fps=10,
    )
    render_video(
        raw_frames,
        output_path=CLIP_OUTPUT + "/{}_{}_raw.mp4".format(IDX,tag),
        fps=10,
    )
    print("Saved {}.".format(tag))


def realtime_object_tracking():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Real-time tracking the object using tactile sensors. Modified to record contact frame deltas and gx/gy gradient values. Output saved to visualization_results folder."
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
        #device_name = input("Type device name (i.e. DIGIT or GelSight Mini)") TODO: Delete
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
    print("\nStart object tracking. Press 'r' to start/stop recording, 'q' to quit.\n")
    is_running = True
    is_recording = False
    clip_index = 1
    any_clip_recorded = False
    frames = []
    raw_frames = []

    while is_running:
        image = device.get_image()
        G, H, C = recon.get_surface_info(image, ppmm)
        if is_recording:
            frames.append((G, H, C))
            raw_frames.append(image)
        show_frame_count(is_recording, len(frames))

        frame = Frame(G, H, C)
        if not frame.is_contacted:
            resize_show(image, is_recording=is_recording)
            key = cv2.waitKey(1)
            is_recording, should_quit, just_stopped = handle_key(key, is_recording)
            if just_stopped:
                save_clip(frames, raw_frames, bg_image, clip_index)
                clip_index += 1
                any_clip_recorded = True
                frames = []
                raw_frames = []
            if should_quit:
                is_running = False
            continue
        else:
            try:
                # Tracking a new object, wait 2 frames for the contact to stabilize
                for _ in range(2):
                    image = device.get_image()
                    if is_recording:
                        raw_frames.append(image)
                    show_frame_count(is_recording, len(raw_frames))
                    resize_show(image, is_recording=is_recording)
                    key = cv2.waitKey(1)
                    is_recording, should_quit, just_stopped = handle_key(key, is_recording)
                    if just_stopped:
                        save_clip(frames, raw_frames, bg_image, clip_index)
                        clip_index += 1
                        any_clip_recorded = True
                        frames = []
                        raw_frames = []
                    if should_quit:
                        is_running = False
                        break
                    if frame.is_contacted:
                        break
                # Get the surface information of the reference frame (key frame)
                image_start = device.get_image()
                G_start, H_start, C_start = recon.get_surface_info(image_start, ppmm)
                if is_recording:
                    frames.append((G_start, H_start, C_start))
                    raw_frames.append(image_start)

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
                while is_tracking:
                    # Get the surface information of the current frame
                    image_curr = device.get_image()
                    G_curr, H_curr, C_curr = recon.get_surface_info(image_curr, ppmm)
                    if is_recording:
                        frames.append((G_curr, H_curr, C_curr))
                        raw_frames.append(image_curr)
                    show_frame_count(is_recording, len(frames))
                    frame_curr = Frame(G_curr, H_curr, C_curr)
                    if not frame_curr.is_contacted:
                        is_tracking = False
                        break

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
                        )
                        frame_prev = frame_curr
                        prev_T_ref = curr_T_ref
                    except LoseTrackError:
                        # Reset reference frame as the previous frame
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
                            frame_prev = frame_curr
                            prev_T_ref = curr_T_ref
                        except LoseTrackError:
                            # Lose track, set current frame as new start frame
                            print("Lose Track!")
                            is_tracking = False
                            break
                    
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
                    resize_show(
                        cv2.hconcat([image_l, image_r]), is_recording=is_recording
                    )
                    key = cv2.waitKey(1)
                    is_recording, should_quit, just_stopped = handle_key(key, is_recording)
                    if just_stopped:
                        save_clip(frames, raw_frames, bg_image, clip_index)
                        clip_index += 1
                        any_clip_recorded = True
                        frames = []
                        raw_frames = []
                    if should_quit:
                        is_tracking = False
                        is_running = False
            except Exception as e:
                if e == ZeroDivisionError:
                    print("Contact disrupted during contact confirmation loop: {}.".format(e))
                else:
                    print("Reconstruction error: {}".format(e))
                if is_recording:
                    frames.append(("CRASH", str(e), None))  # sentinel
                pass

    device.release()
    cv2.destroyAllWindows()
    print()  # move off the in-place frame-count line

    # If we quit while a recording was still in progress (never toggled off),
    # save whatever was captured as one final clip so it isn't lost.
    if is_recording and frames:
        save_clip(frames, raw_frames, bg_image, clip_index)
        any_clip_recorded = True

    if not any_clip_recorded:
        print("No clips were recorded (recording was never started).")

if __name__ == "__main__":
    realtime_object_tracking()