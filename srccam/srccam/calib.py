from typing import Any

import numpy as np


class Calib:
    """
    Calibration parameters loaded from a YAML file.

    K - intrinsics, D - distortion, r - rotation, t - translation.
    """

    def __init__(self, calib_dict: dict[str, Any]):
        self.cam_to_vr = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
        try:
            self.K = np.asarray(calib_dict["K"], dtype=float)
            self.D = np.asarray(calib_dict["D"], dtype=float)
            angles = np.asarray(calib_dict["r"], dtype=float).reshape(-1)
            roll, pitch, yaw = angles[:3]
            self.r = self.rotation_matrix_from([roll, pitch, yaw]).T
            self.t = np.asarray(calib_dict["t"], dtype=float).reshape(3, 1)
        except KeyError as exc:
            raise CalibInitExceltion(f"Bad calib_dict, initialization failed, calid_dict: {calib_dict}") from exc

    @staticmethod
    def rotation_matrix_from(angles: list):
        sinuses = np.sin(angles)
        cosines = np.cos(angles)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, cosines[0], -sinuses[0]],
                [0, sinuses[0], cosines[0]],
            ],
            dtype=float,
        )
        Ry = np.array(
            [
                [cosines[1], 0, sinuses[1]],
                [0, 1, 0],
                [-sinuses[1], 0, cosines[1]],
            ],
            dtype=float,
        )
        Rz = np.array(
            [
                [cosines[2], -sinuses[2], 0],
                [sinuses[2], cosines[2], 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        return Rz @ Ry @ Rx


class CalibInitExceltion(Exception):
    ...
