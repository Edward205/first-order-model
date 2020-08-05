import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
#from IPython.display import HTML
import warnings
import cv2
import tkinter as tk
warnings.filterwarnings("ignore")

#Load driving video and source image
source_image = imageio.imread('first-order-motion-model/02.png')
driving_video = imageio.mimread('first-order-motion-model/04.mp4')

#Resize image and video to 256x256
source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

#Create a model and load checkpoints
from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                            checkpoint_path='first-order-motion-model/vox-cpk.pth.tar')

#Perform image animation
from demo import make_animation
from skimage import img_as_ubyte

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                            checkpoint_path='first-order-motion-model/vox-cpk.pth.tar')
from demo import make_animation
from skimage import img_as_ubyte

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave('generated.mp4', [img_as_ubyte(frame) for frame in predictions])
