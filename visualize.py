#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml

from laserscan import LaserScan
from laserscanvis import LaserScanVis

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--arch', '-ac',
        type=str,
        required=False,
        default="config/darknet53.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data', '-dc',
        type=str,
        required=False,
        default="config/slam-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        required=False,
        help='Sequence to start. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("ArchConfig", FLAGS.arch)
    print("DataConfig", FLAGS.data)
    print("Sequence", FLAGS.sequence)
    print("offset", FLAGS.offset)
    print("*" * 80)

    # open config file
    try:
        print("Opening arch config file %s" % FLAGS.arch)
        ARCHCFG = yaml.safe_load(open(FLAGS.arch, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    try:
        print("Opening data config file %s" % FLAGS.data)
        DATACFG = yaml.safe_load(open(FLAGS.data, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # create a scan
    # project all opened scans to spheric proj
    sensor = ARCHCFG['dataset']['sensor']
    scan = LaserScan(project=True,
                     H=sensor["img_prop"]["height"],
                     W=sensor["img_prop"]["width"],
                     fov_up=sensor["fov_up"],
                     fov_down=sensor["fov_down"])

    # create a visualizer
    vis = LaserScanVis(scan=scan,
                       scan_names=scan_names,
                       offset=FLAGS.offset)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()
