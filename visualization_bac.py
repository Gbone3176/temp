import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from typing import Union
import copy
import random


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, window_name='3d'):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [1, 0, 0, np.pi / 2])  # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self.rotation = False

    def reset(self):
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [1, 0, 0, np.pi / 2])
        self.orbit(0, 700)

    def close(self):
        cv2.destroyAllWindows()

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        up = self.rot.as_matrix()[:3, 2]
        rotvec_x = up * np.radians(0.01 * self.radius * dx)
        rotvec_y = side * np.radians(0.01 * self.radius * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

    @property
    def param(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        K = np.array([focal, 0., self.W / 2, 0., focal, self.H / 2, 0., 0., 1.]).reshape(3, 3)
        pose = self.pose
        R = self.rot.as_matrix()
        T = -R.transpose() @ pose[:3, 3:]
        camera = {
            "K": K.astype('float32'),
            "R": R,
            'T': T.astype('float32'),
            'dist': np.zeros((1, 5)).astype('float32')
        }
        camera['invK'] = np.linalg.inv(camera['K'])
        camera['P'] = camera['K'] @ np.hstack((camera['R'], camera['T']))
        return camera

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """
        project 3d points to 2d points
        Args:
            points: np.ndarray, [n_points, 3]

        Returns:
            points2d: np.ndarray, [n_points, 2]
        """
        K = self.param['K']
        R = self.param['R']
        T = self.param['T']
        dist = self.param['dist']
        points2d, _ = cv2.projectPoints(points.astype("float32"), R, T, K, dist)

        return points2d.reshape(-1, 2).astype("float32")

    def draw_pose(self, image, player_pose: np.ndarray, color=(0, 0, 255), x_shift=7.7, y_shift=4.05):
        """
        draw 3d pose on image
        Args:
            image: np.ndarray, [h, w, c]
            player_pose: np.ndarray, [25, 3]
            color: tuple, (r, g, b),
            x_shift: float, shift x axis #TODO 偏移取决与外参，需要统一
            y_shift: float, shift y axis
        """
        player_pose = player_pose.copy()
        player_pose[:, 0] -= x_shift
        player_pose[:, 1] -= y_shift
        img = visualize_joints3d(player_pose[None], self.param, image,
                                 camera_width=1920, camera_height=1080,
                                 width=1920, height=1080, convention='body25', color=color)[0][0]
        return img

    def draw_ball_point(self, image, ball_3d: np.ndarray, court_center_3d: np.ndarray, color=(0, 255, 255), radius=5,
                        thickness=5):
        ball_project = self.project_points(ball_3d[0][:3] - court_center_3d)[0]
        cv2.circle(image, (int(ball_project[0]), int(ball_project[1])), radius, color, thickness)
        return image

    def show(self, image, window_name='3d'):
        if self.rotation:
            self.orbit(5, 0)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1)
        if key == ord('w'):
            self.orbit(0, 5)
        elif key == ord('s'):
            self.orbit(0, -5)
        elif key == ord('a'):
            self.orbit(-5, 0)
        elif key == ord('d'):
            self.orbit(5, 0)
        elif key == ord('z'):
            self.scale(1)
        elif key == ord('x'):
            self.scale(-1)
        elif key == ord('i'):
            self.pan(0, 0, 1)
        elif key == ord('k'):
            self.pan(0, 0, -1)
        elif key == ord('j'):
            self.pan(0, 1, 0)
        elif key == ord('l'):
            self.pan(0, -1, 0)
        elif key == ord('q'):
            return False
        elif key == ord('r'):
            self.reset()
        elif key == ord(' '):
            self.rotation = not self.rotation
        return True


def camera_project(points: np.ndarray, camera: dict) -> np.ndarray:
    """
    project 3d points to 2d points
    Args:
        points: np.ndarray, [n_points, 3]
        camera: {"R":np.ndarray [3,3], "T":np.ndarray [3,1], "K":np.ndarray [3,3], "dist":np.ndarray [1,5]}

    Returns:
        points2d: np.ndarray, [n_points, 2]
    """
    r_mat = camera["R"].astype("float64")
    t_vec = camera["T"].astype("float64")
    K = camera["K"].astype("float64")
    dist = camera["dist"].astype("float64")
    points2d, _ = cv2.projectPoints(points.astype("float64"), r_mat, t_vec, K, dist)

    return points2d.reshape(-1, 2).astype("float64")


s_body25_flip_pairs = np.array(
    [[2, 5], [3, 6], [4, 7], [9, 12], [10, 13], [11, 14], [15, 16], [17, 18], [22, 19], [23, 20], [24, 21]], dtype=int)
s_body25_parent_ids = np.array([0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11],
                               dtype=int)

s_coco_flip_pairs = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=int)
s_coco_parent_ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14], dtype=int)

joints_dict = {
    # 'aqa': (s_aqa_flip_pairs, s_aqa_parent_ids),
    'body25': (s_body25_flip_pairs, s_body25_parent_ids),
    # 'openpose_25_hand': (s_body25hand_flip_pairs, s_body25hand_parent_ids)
    "coco17": (s_coco_flip_pairs, s_coco_parent_ids)
}


def cv_draw_joints(im, kpt, vis, flip_pair_ids, color_left=(255, 0, 0), color_right=(0, 255, 0), radius=2):
    for ipt in range(0, kpt.shape[0]):
        if vis[ipt, 0]:
            cv2.circle(im, (int(kpt[ipt, 0] + 0.5), int(kpt[ipt, 1] + 0.5)), radius, color_left, -1)
    for i in range(0, flip_pair_ids.shape[0]):
        id = flip_pair_ids[i][0]
        if vis[id, 0]:
            cv2.circle(im, (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)), radius, color_right, -1)


def cv_draw_joints_parent(im, kpt, vis, parent_ids, color=(0, 0, 255), thickness=1):
    for i in range(0, len(parent_ids)):
        id = parent_ids[i]
        if vis[id, 0] and vis[i, 0]:
            cv2.line(im, (int(kpt[i, 0] + 0.5), int(kpt[i, 1] + 0.5)), (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)),
                     color, thickness=thickness)


def visualize_joints2d(image: np.ndarray,
                       joints: np.ndarray,
                       convention: str = 'body25',
                       joints_mask: np.ndarray = None,
                       color_left: tuple = (255, 0, 0),
                       color_right: tuple = (0, 255, 0),
                       threshold: float = 0.5,
                       name: str = None,
                       radius: int = 4) -> np.ndarray:
    '''
    plot 2d joints on image
    Args:
        image (np.ndarray): [h, w, c]
        joints (np.ndarray): [n_joints, 2]
        convention (str): 'aqa' or 'openpose_25'
        joints_mask (np.ndarray): [n_joints, 1]
        color_left (tuple): BGR format
        color_right (tuple): BGR format
        radius (int): the radius of joint
        name (str): the name of the person
    Returns:
        image (np.ndarray): [h, w, c]
    '''

    if joints_mask is None:
        if joints.shape[-1] > 2:
            joints_mask = joints[..., 2:] > threshold
        else:
            joints_mask = np.ones_like(joints)
    if convention in joints_dict.keys():
        flip_pairs, parent_ids = joints_dict[convention]
    else:
        raise NotImplementedError
    if len(joints.shape) == 3:  # multi person
        for i in range(joints.shape[0]):
            visualize_joints2d(image, joints[i], convention=convention, joints_mask=joints_mask[i])
        return image
    cv_draw_joints(image, joints, joints_mask, flip_pairs, color_left=color_left, color_right=color_right,
                   radius=radius)
    cv_draw_joints_parent(image, joints, joints_mask, parent_ids)
    if name is not None:
        cv2.putText(image, name, (int(joints[0, 0] + 0.5), int(joints[0, 1] + 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color_left, 2)
    return image


def visualize_joints3d(keypoints_arr: np.ndarray,
                       camera: Union[list, dict],
                       background_arr: Union[list, np.ndarray, None] = None,
                       camera_width: int = None,
                       camera_height: int = None,
                       width: int = None,
                       height: int = None,
                       name: str = None,
                       color: tuple = (0, 0, 255),
                       convention: str = 'body25'):
    """
    visualize 3d joints on background
    Args:

        keypoints_arr: np.ndarray, [n_frame, n_joints, 4]
        camera: dict or list of dict, list means multi-view, camera format: easymocap, {"K":,"invK":,"R":,"T":}
        background_arr: np.ndarray, [n_camera, n_frame, h, w, c] or list of np.ndarray
        background_arr should be multi-view too
        camera_width: camera width for the intri and extri
        camera_height: camera height for the intri and extri
        height: image height
        width: image width
        convention: 'aqa' or 'openpose_25'

    Returns:
        images: np.ndarray, [n_camera, n_frame, h, w, c]
    """
    joints = copy.deepcopy(keypoints_arr)
    assert len(joints.shape) == 3
    assert joints.shape[2] >= 3

    if color is not None:
        color_left = color
        color_right = color[::-1]
    else:
        color_left = (255, 0, 0)
        color_right = (0, 255, 0)

    n_frame, n_kpts, _ = joints.shape

    if isinstance(camera, dict):
        camera = [camera]

    if background_arr is None:
        assert width is not None and height is not None
        background_arr = np.zeros((len(camera), n_frame, height, width, 3), dtype=np.uint8)

    # copy background_arr to [1, 1, h, w, c]
    if len(background_arr.shape) == 3:
        background_arr = np.expand_dims(background_arr, axis=0)
        background_arr = np.expand_dims(background_arr, axis=0)

    assert background_arr.shape[0] == len(camera)
    assert background_arr.shape[1] == n_frame

    if joints.shape[2] > 3:
        joints_score = joints[..., 3:]
        joints = joints[..., :3]
    else:
        joints_score = [None] * joints.shape[0]

    for i_cam, cam in enumerate(camera):  # multi view
        assert isinstance(cam, dict)
        assert "K" in cam.keys()
        assert "R" in cam.keys()
        assert "T" in cam.keys()
        assert "dist" in cam.keys()

        view_points2d = camera_project(joints.reshape((n_frame * n_kpts, 3)), cam)
        view_points2d = view_points2d.reshape((n_frame, n_kpts, 2))

        if camera_width is not None and camera_height is not None:
            view_points2d[..., 0] = view_points2d[..., 0] / camera_width * background_arr.shape[3]
            view_points2d[..., 1] = view_points2d[..., 1] / camera_height * background_arr.shape[2]
        # views_points.append(view)

        for i_frame, joint in enumerate(view_points2d):  # multi frame
            # image = np.zeros((height, width, 3), np.uint8)
            # ignore the joints out of the image
            # if joint.min() < 0 or joints[:, 0].max() > background_arr.shape[3] or joints[:, 1].max() > \
            #         background_arr.shape[2]:
            #     continue
            visualize_joints2d(background_arr[i_cam, i_frame], joint.astype("float32"), convention=convention,
                               joints_mask=joints_score[i_frame], name=name, color_left=color_left,
                               color_right=color_right)
            # multi_frame_images.append(background_arr[i_cam, i_frame])
    return background_arr


def get_random_color(idx):
    """
    get random color
    Args:
        idx: index
    Returns:
        color: (r, g, b)
    """
    random.seed(idx)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color


import pickle
if __name__ == "__main__":
    width, height = 1920, 1080
    root = "/home/PJLAB/guobowen/workspace/ai4tennis-backend/test_data_for_pose_visualize.pkl"
    with open(root, 'rb') as f:
        data_loaded = pickle.load(f)
    print(data_loaded['keypoints_list'][0].shape)
    orbit_r = 20
    orbit_dx = 0
    orbit_dy = 200
    cam = OrbitCamera(width, height, orbit_r, fovy=90)
    back = np.zeros((height, width, 3), dtype=np.uint8)
    cam.orbit(orbit_dx, orbit_dy)

    for i in range(450):  # 遍历所有帧
        image = cam.draw_pose(back.copy(), data_loaded['keypoints_list'][0][i], x_shift=11, y_shift=5)
        frame = image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 显示当前帧
        cv2.imshow(f'Frame', frame)
        cv2.waitKey(25)
        # 按 'q' 键退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
