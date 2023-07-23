# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
import os
import time
import torch
from typing import Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_original_cwd
import time
from functools import partial
import matplotlib.pyplot as plt
import shutil
from util.utils import seed_all_random_engines
from util.load_img_folder import load_and_preprocess_images
from util.geometry_guided_sampling import geometry_guided_sampling
from pytorch3d.vis.plotly_vis import get_camera_wireframe
import subprocess
import tempfile
import gradio as gr

def plot_cameras(ax, cameras, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe().cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles

def create_matplotlib_figure(pred_cameras):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.clear()
    handle_cam = plot_cameras(ax, pred_cameras, color="#FF7D1E")
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    
    return plt

import os
import json
import tempfile
from PIL import Image


def convert_extrinsics_pytorch3d_to_opengl(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert extrinsics from PyTorch3D coordinate system to OpenGL coordinate system.

    Args:
        extrinsics (torch.Tensor): a 4x4 extrinsic matrix in PyTorch3D coordinate system.

    Returns:
        torch.Tensor: a 4x4 extrinsic matrix in OpenGL coordinate system.
    """
    # Create a transformation matrix that flips the Z-axis
    flip_z = torch.eye(4)
    flip_z[2, 2] = -1
    flip_z[0, 0] = -1

    # Multiply the extrinsic matrix by the transformation matrix
    extrinsics_opengl = torch.mm(extrinsics, flip_z)

    return extrinsics_opengl

import json
from typing import List, Dict, Any

def create_camera_json(extrinsics: Any, focal_length_world: float, principle_points: List[float], image_size: int) -> str:
    # Initialize the dictionary
    camera_dict = {
        "w": image_size,
        "h": image_size,
        "fl_x": float(focal_length_world[0]),
        "fl_y": float(focal_length_world[1]),
        "cx": float(principle_points[0]),
        "cy": float(principle_points[1]),
        "k1": 0.0,  # Assuming these values are not provided
        "k2": 0.0,  # Assuming these values are not provided
        "p1": 0.0,  # Assuming these values are not provided
        "p2": 0.0,  # Assuming these values are not provided
        "camera_model": "OPENCV",
        "frames": []
    }

    # Add frames to the dictionary
    for i, extrinsic in enumerate(extrinsics):
        frame = {
            "file_path": f"images/frame_{str(i).zfill(5)}.jpg",
            "transform_matrix": extrinsic.tolist(),
            "colmap_im_id": i
        }
        # Convert numpy float32 to Python's native float
        frame["transform_matrix"] = [[float(element) for element in row] for row in frame["transform_matrix"]]
        camera_dict["frames"].append(frame)

    return camera_dict

def archieve_images_and_transforms(images, pred_cameras, image_size):
    images_array = images.permute(0, 2, 3, 1).cpu().numpy() * 255
    images_pil = [Image.fromarray(image.astype('uint8')) for image in images_array]

    with tempfile.TemporaryDirectory() as temp_dir:
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        images_path = []
        for i, image in enumerate(images_pil):
            image_path = os.path.join(images_dir, 'frame_{:05d}.jpg'.format(i))
            image.save(image_path)
            images_path.append(image_path)
        
        cam_trans = pred_cameras.get_world_to_view_transform()
        extrinsics = cam_trans.inverse().get_matrix().cpu()
        extrinsics = [convert_extrinsics_pytorch3d_to_opengl(extrinsic.T) for extrinsic in extrinsics]
        
        focal_length_ndc  = pred_cameras.focal_length.mean(dim=0).cpu().numpy()
        focal_length_world = focal_length_ndc * image_size / 2
        principle_points = [image_size / 2, image_size / 2]
        camera_dict = create_camera_json(extrinsics, focal_length_world, principle_points, image_size)

        json_path = os.path.join(temp_dir, 'transforms.json')
        with open(json_path, 'w') as f:
            json.dump(camera_dict, f, indent=4)

        project_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.make_archive(f'/tmp/{project_name}', 'zip', temp_dir)
    return f'/tmp/{project_name}.zip'
        
def estimate_images_pose(image_folder, mode) -> None:
    print("Slected mode:", mode)
    with hydra.initialize(config_path="./cfgs/"):
        cfg = hydra.compose(config_name=mode)
        
    OmegaConf.set_struct(cfg, False)
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)

    # Load and preprocess images
    images, image_info = load_and_preprocess_images(image_folder, cfg.image_size)

    # Load checkpoint
    ckpt_path = os.path.join(cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")

    # Move model and images to the GPU
    model = model.to(device)
    images = images.to(device)

    # Evaluation Mode
    model.eval()

    # Seed random engines
    seed_all_random_engines(cfg.seed)

    # Start the timer
    start_time = time.time()

    # Perform match extraction
    cond_fn = None
    print("[92m=====> Sampling without GGS <=====[0m")

    # Forward
    with torch.no_grad():
        # Obtain predicted camera parameters
        # pred_cameras is a PerspectiveCameras object with attributes
        # pred_cameras.R, pred_cameras.T, pred_cameras.focal_length

        # The poses and focal length are defined as
        # NDC coordinate system in
        # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        pred_cameras = model(
            image=images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step
        )

    # Stop the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken: {:.4f} seconds".format(elapsed_time))

    zip_path = archieve_images_and_transforms(images, pred_cameras, cfg.image_size)
    return create_matplotlib_figure(pred_cameras), zip_path

def extract_frames_from_video(video_path: str) -> str:
    """
    Extracts frames from a video file and saves them in a temporary directory.
    Returns the path to the directory containing the frames.
    """
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%03d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "fps=1",
        output_path
    ]
    subprocess.run(command, check=True)
    return temp_dir

def estimate_video_pose(video_path: str, mode: str) -> plt.Figure:
    """
    Estimates the pose of objects in a video.
    """
    # Extract frames from the video
    image_folder = extract_frames_from_video(video_path)
    # Estimate the pose for each frame
    fig = estimate_images_pose(image_folder, mode)
    return fig

if __name__ == "__main__":
    examples = [["examples/" + img, 'fast'] for img in os.listdir("examples/")]
    # Create a Gradio interface
    iface = gr.Interface(
        fn=estimate_video_pose,
        inputs=[gr.inputs.Video(label='video', type='mp4'),
                gr.inputs.Radio(choices=['fast', 'precise'],  default='fast',
                                label='Estimation Model, fast is quick, usually within 1 seconds; precise has higher accuracy, but usually take several minutes')],
        outputs=['plot', 'file'],
        title="PoseDiffusion Demo: Solving Pose Estimation via Diffusion-aided Bundle Adjustment",
        description="Upload a video for object pose estimation. The object should be centrally located within the frame.",
        examples=examples,
        cache_examples=True
    )
    iface.launch()