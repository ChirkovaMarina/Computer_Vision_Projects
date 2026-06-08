import os
import gzip
import yaml
import math

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import cv2
import numpy as np

from srccam import CalibReader, Calib, Camera, SeasonReader, Point3d as Point


class PathCreator:
    """Compute and render the tram path from GPS data."""

    # Paths to the metadata archives.
    DATA_PATHS = ['../data/city/trm.169.007.info.yml.gz',
                  '../data/city/trm.169.008.info.yml.gz']

    GPS_TAG = 'emlidLeft'  # Key used to access GPS data.
    POINT_CNT = 50         # Number of points to draw.
    COLOR = (0, 0, 255)    # Path color (red).
    LINE_WIDTH = 5         # Line thickness.

    def __init__(self, calib) -> None:
        self.camera = Camera(calib)

        # GPS data.
        self.data = self.get_gps_data()

        # Distances and azimuths for adjacent path points.
        self.dists, self.angles = np.zeros(len(self.data) - 1), np.zeros(len(self.data) - 1)

        # Convert raw data to distances and angles between adjacent points.
        self.process_data()

        # Current tram heading angle.
        self.zero_angle = 0

        # Current point index in the path.
        self.current_index = 0

        # Current path segment.
        self.path = [(self.dists[i], self.angles[i]) for i in range(self.POINT_CNT)]

        # 2D points used to draw the path.
        self.points_2d = None

    def get_gps_data(self) -> list:
        """Load tram path information from metadata archives."""

        data = []
        for file_path in self.DATA_PATHS:
            if not os.path.isfile(file_path):
                raise FileNotFoundError('File {} not found.'.format(file_path))

            with gzip.open(file_path, 'rt') as file:
                file.readline()
                file = file.read()
                file = file.replace(':', ': ')
                yml = yaml.load(file, Loader=Loader)
                for shot in yml['shots']:
                    if self.GPS_TAG in shot:
                        sense_data = shot[self.GPS_TAG]['senseData']
                        data.append({'nord': sense_data['nord'],
                                     'east': sense_data['east']})

        return data

    def process_data(self) -> None:
        """Compute distances and azimuths for all known points."""

        for i in range(len(self.data) - 1):
            p1, p2 = self.data[i], self.data[i + 1]
            self.dists[i], self.angles[i] = self.get_dist_and_angle(p1['nord'], p1['east'], p2['nord'], p2['east'])

    @staticmethod
    def get_dist_and_angle(lat1: float, long1: float, lat2: float, long2: float) -> (float, float):
        """Compute the distance and initial azimuth between two coordinates."""

        # Earth radius.
        rad = 6372795

        # Convert coordinates to radians.
        lat1 = lat1 * math.pi / 180
        lat2 = lat2 * math.pi / 180
        long1 = long1 * math.pi / 180
        long2 = long2 * math.pi / 180

        # Cosines and sines of latitudes and longitude differences.
        cl1 = math.cos(lat1)
        cl2 = math.cos(lat2)
        sl1 = math.sin(lat1)
        sl2 = math.sin(lat2)
        delta = long2 - long1
        c_delta = math.cos(delta)
        s_delta = math.sin(delta)

        # Compute the distance between points.
        y = math.sqrt(math.pow(cl2 * s_delta, 2) + math.pow(cl1 * sl2 - sl1 * cl2 * c_delta, 2))
        x = sl1 * sl2 + cl1 * cl2 * c_delta
        ad = math.atan2(y, x)
        dist = ad * rad

        # Compute the initial azimuth.
        y = s_delta * cl2
        x = (cl1 * sl2) - (sl1 * cl2 * c_delta)
        angle_rad = math.atan2(y, x)

        return dist, angle_rad

    def next_point(self) -> None:
        """Append the next point to the route."""

        self.current_index += 1
        self.path = self.path[1:]
        point_index = min(self.current_index + self.POINT_CNT, len(self.dists) - 1)
        self.path.append((self.dists[point_index], self.angles[point_index]))
        self.points_2d = None

    def render_path(self, frame) -> None:
        """Render the path on the frame."""

        if self.points_2d is None:  # Rebuild if the previous projected path does not exist or is stale.
            # Points in 3D space.
            points_3d = [Point((0, 0, 0))]

            # Compute the current tram heading angle.
            self.zero_angle = 0
            for i in range(10):
                index = max(self.current_index + i - 24, 0)
                self.zero_angle += self.angles[index]
            self.zero_angle /= 10
            # Use the average heading from several previous path segments.

            # Compute 3D points used to draw the path.
            x, y = 0, 0
            for (dist, angle) in self.path:
                angle -= self.zero_angle
                x += dist * math.sin(angle)
                y += dist * math.cos(angle)
                points_3d.append(Point((x, y, 0)))

            # Project points into 2D image space.
            self.points_2d = np.array([self.camera.project_point_3d_to_2d(point) for point in points_3d], np.int32)

        # Draw the path.
        cv2.polylines(frame, [self.points_2d], False, self.COLOR, self.LINE_WIDTH, cv2.LINE_AA)


class PathPredictor(SeasonReader):
    """Video stream processing."""

    def __init__(self) -> None:
        self.path_creator = None

    def on_init(self) -> bool:
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(file_name='../data/city/leftImage.yml', param=par)
        calib_dict = calib_reader.read()
        calib = Calib(calib_dict)
        self.path_creator = PathCreator(calib)
        return True

    def on_shot(self) -> bool:
        return True

    def on_frame(self) -> bool:
        self.path_creator.render_path(self.frame)
        return True

    def on_gps_frame(self) -> bool:
        self.path_creator.next_point()
        return True

    def on_imu_frame(self) -> bool:
        return True


if __name__ == '__main__':
    predictor = PathPredictor()
    predictor.initialize(path_to_data_root='../data/city/')
    try:
        predictor.run()
    except FileNotFoundError as e:
        print('FileNotFoundError: {}.'.format(e))
    print('Done!')
