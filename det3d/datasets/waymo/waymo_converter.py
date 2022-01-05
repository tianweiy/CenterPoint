"""Tool to convert Waymo Open Dataset to pickle files.
    Adapted from https://github.com/WangYueFt/pillar-od
    # Copyright (c) Massachusetts Institute of Technology and its affiliates.
    # Licensed under MIT License
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, argparse, tqdm, pickle, os 

import waymo_decoder 
import tensorflow.compat.v2 as tf
from waymo_open_dataset import dataset_pb2

from multiprocessing import Pool 

tf.enable_v2_behavior()

fnames = None 
LIDAR_PATH = None
ANNO_PATH = None 

def convert(idx):
    global fnames
    fname = fnames[idx]
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    for frame_id, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        decoded_frame = waymo_decoder.decode_frame(frame, frame_id)
        decoded_annos = waymo_decoder.decode_annos(frame, frame_id)

        with open(os.path.join(LIDAR_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_frame, f)
        
        with open(os.path.join(ANNO_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_annos, f)


def main(args):
    global fnames 
    fnames = sorted(list(glob.glob(args.record_path)))

    print("Number of files {}".format(len(fnames)))

    with Pool(128) as p: # change according to your cpu
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--record_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    LIDAR_PATH = os.path.join(args.root_path, 'lidar')
    ANNO_PATH = os.path.join(args.root_path, 'annos')

    if not os.path.isdir(LIDAR_PATH):
        os.mkdir(LIDAR_PATH)

    if not os.path.isdir(ANNO_PATH):
        os.mkdir(ANNO_PATH)
    
    main(args)
