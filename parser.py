import os
import numpy as np
import torch
from torch.utils.data import Dataset
from laserscan import LaserScan
from poseloader import PoseLoader
from scipy.spatial.transform import Rotation as R

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_POSE = ['.txt']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_pose(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_POSE)


class SlamKitti(Dataset):

    def __init__(self, root,    # directory where data is
                 sequences,     # sequences for this data (e.g. [1,3,4,6])
                 sensor,              # sensor to parse scans from
                 max_points=150000,   # max number of points present in dataset
                 gt=True):            # send ground truth?
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure sequences is a list
        assert(isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.pose_files = []
        self.sequences_scan_num = [0]

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            pose_path = os.path.join(self.root, seq, "pose_6dof")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            pose_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(pose_path)) for f in fn if is_pose(f)]

            # check all scans have labels
            if self.gt:
                assert(len(scan_files) == len(pose_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.pose_files.extend(pose_files)

            self.sequences_scan_num.append(
                self.sequences_scan_num[-1] + len(scan_files))

        # sort for correspondance
        self.scan_files.sort()
        self.pose_files.sort()

        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))

    def getSingleItem(self, index):
      # get item in tensor shape
        scan_file = self.scan_files[index]
        if self.gt:
            pose_file = self.pose_files[index]

        # open a laserscan
        scan = LaserScan(project=True,
                         H=self.sensor_img_H,
                         W=self.sensor_img_W,
                         fov_up=self.sensor_fov_up,
                         fov_down=self.sensor_fov_down)

        pose = PoseLoader()
        # open and obtain scan
        scan.open_scan(scan_file)
        if self.gt:
            pose_vec = pose.open_pose(pose_file)
        else:
            pose_vec = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)

        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()
        return proj, pose_vec

    def __getitem__(self, index):
        for seq_i in self.sequences_scan_num:
            if index == seq_i:
                index = index + 1
                break
        scan0, pose0 = self.getSingleItem(index - 1)
        scan1, pose1 = self.getSingleItem(index)

        Rm0 = R.from_rotvec(pose0[3:])
        Rm1 = R.from_rotvec(pose1[3:])
        Rm0_inv = Rm0.inv()
        t0_inv = np.diagonal(-(Rm0_inv.as_matrix() * pose0[0:3]), offset=0)
        delta_t = t0_inv + \
            np.diagonal(Rm0_inv.as_matrix() * pose1[0:3], offset=0)
        delta_r = (Rm0_inv*Rm1).as_rotvec()
        delta_pose = np.concatenate([delta_t, delta_r])

        # return
        return scan0, scan1, delta_pose

    def __len__(self):
        return len(self.scan_files) - 1


class Parser():
    # standard conv, BN, relu
    def __init__(self,
                 root,              # directory for data
                 train_sequences,   # sequences to train
                 valid_sequences,   # sequences to validate.
                 test_sequences,    # sequences to test (if none, don't get)
                 sensor,            # sensor to use
                 max_points,        # max points in each scan in entire dataset
                 batch_size,        # batch size for train and val
                 workers,           # threads to load data
                 gt=True,           # get gt?
                 shuffle_train=True):  # shuffle training set?
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train

        # Data loading code
        self.train_dataset = SlamKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=self.shuffle_train,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

        self.valid_dataset = SlamKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)

        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

        if self.test_sequences:
            self.test_dataset = SlamKitti(root=self.root,
                                          sequences=self.test_sequences,
                                          sensor=self.sensor,
                                          max_points=max_points,
                                          gt=False)

            self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.workers,
                                                          pin_memory=True,
                                                          drop_last=True)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)
