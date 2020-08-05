import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
#from IPython.display import HTML
import warnings
import cv2
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings("ignore")

#Load driving video and source image
def generate():
    source_image = imageio.imread(image_path)
    driving_video = imageio.mimread(video_path)
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
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions])
    print('video saved at ' + output_path)

def pickImage():
    global image_path
    image_path = filedialog.askopenfilename(initialdir = "./", title = "Select a File", filetypes=[("Image", ".png .jpg .jpeg")])

def pickVideo():
    global video_path
    video_path = filedialog.askopenfilename(initialdir = "./", title = "Select a File", filetypes=[("Video", ".mp4 .mpeg .avi")])

def pickFolder():
    global output_path
    output_path = filedialog.askdirectory() + ".mp4"
def start():
    generate()

main = tk.Tk()
pick_image = tk.Button(main, text ="Pick image", padx = 10, command = pickImage)
pick_video = tk.Button(main, text ="Pick driving video", padx = 10, command = pickVideo)
pick_gen_location = tk.Button(main, text ="Pick output path", padx = 10, command = pickFolder)
start = tk.Button(main, text ="Generate video", padx = 10, command = start)

pick_image.pack()
pick_video.pack()
pick_gen_location.pack()
start.pack()
main.mainloop()
