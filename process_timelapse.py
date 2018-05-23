

import os
import sys

from scipy.misc import imread

from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input


cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Directory to read images from")
parser.add_argument("-o", "--output", help="should end in .gif")
args = parser.parse_args()

import os
from PIL import Image
import numpy as np
images = []
for im in sorted(os.listdir(args.input)):
    try:
        print(im)
        images.append(imread(args.input + im, mode='RGB')[::2, ::2])
    except OSError:
        print(im, "was invalid")


poses = []
for image in images:
    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    poses.append(pose)



poses = np.array(poses)


finished = []

for i in range(len(images)):
    fixed = poses[0, (9, 11), 0:2].transpose() 
    moving = poses[i, (9, 11), 0:2].transpose()

    fixedtrans = np.sum(fixed, 1, keepdims=True) / 2
    movingtrans = np.sum(moving, 1, keepdims=True) / 2
    
    trans = fixedtrans-movingtrans
    
    scale = np.abs(fixedtrans) /np.abs(movingtrans)
    print(trans)
    finished.append(np.roll(
        np.roll(images[i], int(trans[0, 0]), 1)
        ,int(trans[1, 0]), 0))
    
    
from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip(finished, fps=12)
clip.write_gif(args.output, fps=12)

