"""Tool to convert Waymo Open Dataset to tf.Examples.
    Taken from https://github.com/WangYueFt/pillar-od
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import glob

import time
import tensorflow.compat.v2 as tf
import waymo_decoder
from waymo_open_dataset import dataset_pb2
import pickle
from multiprocessing import Pool 
import tqdm

tf.enable_v2_behavior()

flags.DEFINE_string('input_file_pattern', None, 'Path to read input')
flags.DEFINE_string('output_filebase', None, 'Path to write output')

FLAGS = flags.FLAGS
fnames = None 

def convert(idx):
  global fnames
  fname = fnames[idx]
  dataset = tf.data.TFRecordDataset(fname, compression_type='')
  for frame_id, data in enumerate(dataset):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    decoded_frame = waymo_decoder.decode_frame(frame)
   
    with open(FLAGS.output_filebase+'seq_{}_frame_{}.pkl'.format(idx, frame_id), 'wb') as f:
      pickle.dump(decoded_frame, f)


def main(unused_argv):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  assert FLAGS.input_file_pattern
  assert FLAGS.output_filebase


  global fnames 
  fnames = list(glob.glob(FLAGS.input_file_pattern))

  print("Number of files {}".format(len(fnames)))

  with Pool(8) as p:
    r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))



if __name__ == '__main__':
  app.run(main)
