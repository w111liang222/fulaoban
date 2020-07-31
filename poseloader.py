#!/usr/bin/env python3
import numpy as np


class PoseLoader:
    """Class that contains LaserScan with dx,dy,dz,rx,ry,rz"""
    EXTENSIONS_POSE = ['.txt']

    def __init__(self):
        pass

    def open_pose(self, filename):
        """ Open pose and fill in attributes
        """

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a pose
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_POSE):
            raise RuntimeError("Filename extension is not valid pose file.")

        # if all goes well, open pose
        pose_vec = np.loadtxt(filename, dtype=np.float32)
        return pose_vec
