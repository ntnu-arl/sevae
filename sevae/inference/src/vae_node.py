import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


import numpy as np
import torch


from sevae.inference.scripts.VAENetworkInterface import VAENetworkInterface

import argparse

# Use argparser to distinguish between simulation and real world
parser = argparse.ArgumentParser()
parser.add_argument('--sim', default="False")
parser.add_argument('--show_cv', default="False")
parser.add_argument('--sim_use_flightmare', default="False")
args = parser.parse_args()

LATENT_SPACE = 128
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 270
IMAGE_DIMENSIONS = IMAGE_WIDTH*IMAGE_HEIGHT

IMAGE_MAX_DEPTH = 10.0 #7.50

LATENT_SPACE_TOPIC_NAME = "/latent_space"

if args.sim == "False":
    IS_SIM = False
    IMAGE_TOPIC = "/d455/depth/image_rect_raw"
    USE_CV_BRIDGE = False
    CALCULATE_RECONSTRUCTION = True

else:
    IS_SIM = True
    print('args.sim_use_flightmare:', args.sim_use_flightmare)
    if args.sim_use_flightmare == "True":
        IMAGE_TOPIC = "/delta/agile_autonomy/unity_depth" # '/delta/agile_autonomy/sgm_depth'
    else: # Gazebo
        IMAGE_TOPIC = "/delta/rgbd/camera_depth/depth"
    print('IMAGE_TOPIC:', IMAGE_TOPIC)
    USE_CV_BRIDGE = True
    CALCULATE_RECONSTRUCTION = True
from cv_bridge import CvBridge, CvBridgeError


class VAENavInterface():
    def __init__(self):

        self.obs_tensor = torch.zeros(
            IMAGE_DIMENSIONS, device="cuda:0")
        self.target_x = 0
        self.target_y = 0
        self.target_z = 0
        self.skip_next = False
        self.net_interface = VAENetworkInterface(LATENT_SPACE, "cuda:0")
        if IS_SIM == True:
            self.bridge = CvBridge()
        # Publish filtered image
        self.decode_img_publisher = rospy.Publisher('/decoded_image', Image, queue_size=1)
        # Subscribe to image
        self.image_sub = rospy.Subscriber(
            IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        print("Subscribed to {}".format(IMAGE_TOPIC))
        
        # Publish latent space as Float32MultiArray
        self.latent_space_publisher = rospy.Publisher(LATENT_SPACE_TOPIC_NAME, Float32MultiArray, queue_size=2)
        print("Publishing latent space to {}", LATENT_SPACE_TOPIC_NAME)


    def image_callback(self, data):
        if self.skip_next:
            self.skip_next = False
            return

        # # convert from ROS image to torch image
        if IS_SIM:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            except CvBridgeError as e:
                print(e)
            cv_image = np.copy(cv_image)
            cv_image = cv_image.reshape((data.height,data.width))
            if args.sim_use_flightmare == "True":
                cv_image = cv_image[5:275,:] # flightmare crashes if we use the resolution of 480x270, hence we use 480x280! 
                cv_image = cv_image.astype('float32') * 0.001 # convert pixel value to meter
            cv_image[np.isnan(cv_image)] = IMAGE_MAX_DEPTH
        else:
            cv_image = np.ndarray((data.height, data.width), '<H', data.data, 0)
            cv_image = cv_image.astype('float32') * 0.001 # convert pixel value to meter
            cv_image[np.isnan(cv_image)] = IMAGE_MAX_DEPTH # max_range


        # send image to GPU on torch
        np_image = np.asarray(cv_image).copy()
        np_image[np_image > IMAGE_MAX_DEPTH] = IMAGE_MAX_DEPTH
        np_image[np_image < 0.20] = -1.0
        
        np_image.setflags(write=1)
        np_image = np_image/IMAGE_MAX_DEPTH
        np_image[np_image < 0.2/IMAGE_MAX_DEPTH] = -1.0

        input_image = np_image.copy()
        if np_image.shape != (IMAGE_HEIGHT,IMAGE_WIDTH):
            torch_img = torch.from_numpy(np_image).float()
            rescaled_image = torch.nn.functional.interpolate(torch_img.unsqueeze(0).unsqueeze(0), (IMAGE_HEIGHT, IMAGE_WIDTH)).squeeze(0).squeeze(0)
            input_image = rescaled_image.numpy()


        means, reconstruction, compute_time = self.net_interface.forward(input_image, calculate_reconstruction=CALCULATE_RECONSTRUCTION)
        
        if CALCULATE_RECONSTRUCTION == True:
            img_filtered_uint8 = (reconstruction * 255).astype(np.uint8)
            msg_filtered = Image()
            msg_filtered.height = IMAGE_HEIGHT # hardcoded and not data.height
            msg_filtered.width = IMAGE_WIDTH # hardcoded and not data.width
            msg_filtered.encoding = "8UC1"
            msg_filtered.is_bigendian = 0
            msg_filtered.step = IMAGE_WIDTH # 1 byte for each pixel
            msg_filtered.data = np.reshape(img_filtered_uint8, (IMAGE_WIDTH*IMAGE_HEIGHT,)).tolist()
            self.decode_img_publisher.publish(msg_filtered)


        latent_space_msg = Float32MultiArray()
        latent_space_msg.data = means.flatten().tolist()
        self.latent_space_publisher.publish(latent_space_msg)
        print('Compute time:', compute_time)


if __name__ == "__main__":
    rospy.init_node("vae_interface")
    print("Node Initialized. Loading weights.")
    nav_interface = VAENavInterface()
    print("Loaded weights. Lets gooooooooooo.......")
    rospy.spin()
