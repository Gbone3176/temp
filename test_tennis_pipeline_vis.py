import os
from typing import List
import cProfile

import cv2
import numpy as np
from backends.models.utils.visualization import OrbitCamera
from backends.models.utils.logger import init_logger
from backends.models.pipelines.player_ball_tracking import PlayerBallTrackingPipeline
from backends.models.pipelines.multiprocess_player_ball_tracking import (
    MultiprocessPlayerBallTrackingPipeline,
)


def load_config(config_path: str):
    """
    load .py config as a module
    """
    import importlib

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    if not config_path.endswith(".py"):
        raise ValueError(f"Config file {config_path} is not a .py file")
    if config_path.startswith("/") or config_path.startswith("\\"):
        raise ValueError(f"Config file {config_path} path should be relative")

    # import the config file as a module
    config_module = importlib.import_module(
        config_path.lstrip(".")
        .lstrip("\\")
        .replace("/", ".")
        .replace("\\", ".")
        .rstrip(".py")
    )
    return config_module


def profile_code(func):
    """
    profile the code
    """
    import pstats
    import io

    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats()
    print(stream.getvalue())

def save_video(images, output_path, fps=30, width=None, height=None):
    # 创建一个 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编解码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        out.write(image)
    out.release()

    # 输出完成信息
    print(f"video saved：{output_path}")

def test_tennis_torch(
        config: str,
        camera_ids: List[str],
        video_path: str,
        camera_path: str,
        save_path: str,
        with_profile: bool = False,
        num_workers: int = 1,
        vis_opt: bool = True,
        save_opt: bool = True,

):
    width, height = 1920, 1080
    orbit_r = 20
    orbit_dx = 0
    orbit_dy = 200
    cam = OrbitCamera(width, height, orbit_r, fovy=90)
    back = np.zeros((height, width, 3), dtype=np.uint8)
    cam.orbit(orbit_dx, orbit_dy)


    logger = init_logger(name="Test")
    config_module = load_config(config)
    logger.info(f"Loading config from {config}")
    video_path_dict = {
        cid: os.path.join(video_path, f"{cid}.mp4") for cid in camera_ids
    }
    intri_path = os.path.join(camera_path, "intri.yml")
    extri_path = os.path.join(camera_path, "extri.yml")
    if num_workers > 1:
        pipeline = MultiprocessPlayerBallTrackingPipeline(
            court_cfg=config_module.court_cfg,
            player_tracker_cfg=config_module.player_tracker_cfg,
            ball_tracker_cfg=config_module.ball_tracker_cfg,
            triangularizator_cfg=config_module.triangularizator_cfg,
            video_load_cfg=config_module.video_load_cfg,
            num_workers=num_workers,
        )
    else:
        pipeline = PlayerBallTrackingPipeline(
            court_cfg=config_module.court_cfg,
            player_tracker_cfg=config_module.player_tracker_cfg,
            ball_tracker_cfg=config_module.ball_tracker_cfg,
            triangularizator_cfg=config_module.triangularizator_cfg,
            video_load_cfg=config_module.video_load_cfg,
        )
    logger.info(f"Pipeline initialized")

    def run_pipeline():
        images = []
        for frame in pipeline.run(video_path_dict, intri_path, extri_path, camera_ids):
            logger.info(f"Frame {frame.frame_id} processed, with {len(frame.players_3d)} players")
            if vis_opt :
                logger.info(f"vis Frame {frame.frame_id}")
                image = back.copy()
                if len(frame.players_3d) == 2:
                    image = cam.draw_pose(image, frame.players_3d[0].pose_3d, x_shift=11, y_shift=5)
                    image = cam.draw_pose(image, frame.players_3d[1].pose_3d, x_shift=11, y_shift=5)
                elif len(frame.players_3d) == 1:
                    image = cam.draw_pose(image, frame.players_3d[0].pose_3d, x_shift=11, y_shift=5)

                if frame.ball_3d is not None:
                    image = cam.draw_ball_point(image, frame.ball_3d.pose_3d, frame.court.court_center_3d)
                cam.show(image)
                images.append(image)

        if save_opt:
            save_video(images, save_path, fps=30, width=width, height=height)

    if with_profile:
        profile_code(run_pipeline)
    else:
        run_pipeline()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="backends/models/configs/tennis_trt.py"
    )
    parser.add_argument(
        "--camera_ids",
        nargs="+",
        type=str,
        default=["01", "02", "03", "04", "05", "06"],
    )
    parser.add_argument(
        "--video_path", type=str, default="data/demo_videos/0903_clip_01"
    )
    parser.add_argument(
        "--camera_path", type=str, default="data/demo_videos/0903_clip_01"
    )
    parser.add_argument(
        "--save_path", type=str, default="tests/models/01output.mp4"
    )
    parser.add_argument("--with_profile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--vis_opt", type=bool, default=True)
    parser.add_argument("--save_opt", type=bool, default=True)

    args = parser.parse_args()

    if args.num_workers > 1:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")



    test_tennis_torch(
        args.config,
        args.camera_ids,
        args.video_path,
        args.camera_path,
        args.save_path,
        args.with_profile,
        args.num_workers,
        args.vis_opt,
        args.save_opt,
    )
