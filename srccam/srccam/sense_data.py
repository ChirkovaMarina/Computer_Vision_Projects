from math import pi


class SenseData:
    # General information
    _grabNumber: int = 0
    _timestamp: str = ""
    # Geographic position in degrees
    _nord: float = 0.0  # Latitude
    _east: float = 0.0  # Longitude
    _alt: float = 0.0  # Altitude above sea level
    # Speed in m/s
    _speed: float = 0.0
    # Yaw angle in radians
    _yaw: float = 0.0

    def __init__(self, data_) -> None:
        self._grabNumber = data_["grabNumber"]
        data_ = data_["senseData"]
        self._timestamp = data_["timestamp"]
        self._nord = data_["nord"]
        self._east = data_["east"]
        self._alt = data_["alt"]
        self._speed = data_["speed"]
        self._yaw = data_["yaw"]

    # Convert decimal degrees to degrees, minutes and seconds.
    @staticmethod
    def _dd2dms(dd) -> dict:
        grad = int(dd)
        minute = dd * 60 - grad * 60
        sec = minute * 60 - int(minute) * 60
        minute = int(minute)
        sec = int(sec)
        return {"grad": grad, "minute": minute, "sec": sec}

    def convert_geo(self) -> dict:
        return {"nord": self._dd2dms(self._nord), "east": self._dd2dms(self._east)}

    def get_geo_str(self):
        geo = self.convert_geo()
        nord_str = (
            str(geo["nord"]["grad"])
            + "D"
            + str(geo["nord"]["minute"])
            + "'"
            + str(geo["nord"]["sec"])
            + '"'
        )
        east_str = (
            str(geo["east"]["grad"])
            + "D"
            + str(geo["east"]["minute"])
            + "'"
            + str(geo["east"]["sec"])
            + '"'
        )
        return nord_str, east_str

    # Convert yaw from radians to degrees.
    def rad2deg(self) -> float:
        degrees = self._yaw * 180 / pi
        if degrees < 0:
            degrees = 360 + degrees
        degrees = round(degrees, 2)
        return degrees

    def ms2kmh(self) -> float:
        return round(self._speed * 3.6, 3)
