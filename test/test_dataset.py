from pygments.lexer import include

from helper.rmrc import Rmrc
import numpy as np
from scipy.misc import imread
import os
import h5py
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
from scipy.misc import imrotate
import torch
from torch.utils.data import DataLoader
from helper.visualizer import save_result, view_result, plot_loss
from helper.augmentation import *
import random as rnd

# comment this path and create a new vaiable according to your own path.
rmrc_data_path = "/home/sumit/Documents/repo/datasets/rmrc/train"
#rmrc_data_path = "/home/neha/Documents/TUM_Books/projects/dlcv_proj/data/rmrc/train"


def load_rmrc():
    print("loading rmrc...")
    files = os.listdir(rmrc_data_path)
    for file in files:
        file_path = os.path.join(rmrc_data_path, file)
        print(file_path)

    new_file = open("/home/sumit/Documents/repo/datasets/rmrc/depth2", "rb")
    #rmrc_data = h5py.File("/home/sumit/Documents/repo/datasets/rmrc/dataset_rgb2depth_train_part1_of_14.mat")
    rmrc_data = pickle.load(new_file)
    print(len(rmrc_data.items()))
    data = {}
    for k, v in rmrc_data.items():
        print(k)
        data[k] = np.array(v)
    #new_file = open("/home/sumit/Documents/repo/datasets/rmrc/depth2", "wb")
    #pickle.dump(data, new_file)
    #byte_data = bytes(new_file)
    #new_file.write(byte_data)
    #np.save("/home/sumit/Documents/repo/datasets/rmrc/depth1", data['depths'][0])
    print(data['heights'])
    print(np.shape(data['depths'][0]))
    print(np.shape(data['images'][0]))
    print(data['depths'].dtype)
    print(data['images'].dtype)
    x = data['depths'][0]
    y = data['images'][0]
    plt.imshow(x)
    plt.show()
    y = y.transpose(1,2,0)
    plt.imshow(y)
    plt.show()

def augment_data():
    with open("/home/sumit/Documents/repo/datasets/rmrc/train/data_1", "rb") as file:
        data = pickle.load(file)
        N, _, _, _ = np.shape(data['images'])
        images = data['images']
        depths = data['depths']
        augmented_data = {}
        augmented_images = []
        augmented_depths = []
        for i in range(N):
            img = images[i]
            depth = depths[i]
            img = img.transpose(1, 2, 0)
            plt.imshow(img)
            plt.show()
            print(np.shape(depth))

            img_rot90 = imrotate(img, 90)
            depth_rot90 = np.rot90(depth, 1)
            plt.imshow(img_rot90)
            plt.show()
            print(np.shape(depth_rot90))

            img_rot180 = imrotate(img, 180)
            depth_rot180 = np.rot90(depth, 2)
            plt.imshow(img_rot180)
            plt.show()
            print(np.shape(depth_rot180))

            img_rot270 = imrotate(img, 270)
            depth_rot270 = np.rot90(depth, 3)
            plt.imshow(img_rot270)
            plt.show()
            print(np.shape(depth_rot270))

            img_fliplr = np.fliplr(img)
            depth_fliplr = np.fliplr(depth)
            plt.imshow(img_fliplr)
            plt.show()
            print(np.shape(depth_fliplr))

            img_flipud = np.flipud(img)
            depth_flipud = np.flipud(depth)
            plt.imshow(img_flipud)
            plt.show()
            print(np.shape(depth_flipud))

            img_invert = np.invert(img)
            depth_invert = depth
            plt.imshow(img_invert)
            plt.show()
            print(np.shape(depth_invert))
            break;
        #augmented_data['images'] = np.array(augmented_images)
        #augmented_data['depths'] = np.array(augmented_depths)
        binary_file = open("/home/sumit/Documents/repo/datasets/rmrc/train/augmented_data_1", "wb")
        pickle.dump(augmented_data, binary_file)

def test_data_loader():
    dataset = Rmrc(data_path = rmrc_data_path)
    loader = DataLoader(dataset, num_workers=1)
    for idx, (images, depths) in enumerate(loader):
        print(np.shape(images))
        #print(np.shape(images[0]))

    #print(np.shape(loader[0]['images']))
    print("loader..")

def test_data_sampling():
    train = Rmrc(data_path = rmrc_data_path)
    for i in range(1):
        images, depths = train.first()
        print(np.max(images))
        print(np.shape(images))
        print(np.shape(depths))
        view_result(images[0], depths[0])
        print(depths[0])
        if depths[0].any() < 0:
            print("--ve")
        depths[0][np.isnan(depths[0])] = -1
        view_result(images[0], depths[0])
        print(depths[0])
        # images = images[0].transpose(1, 2, 0)
        # flipped = invert_colors(images).transpose(2, 0, 1)
        # view_result(flipped, invert_colors(depths[0]))
        """
        images = images[0].transpose(1, 2, 0)
        plt.imshow(images)
        plt.show()
        plt.imshow(invert_colors(images))
        plt.show()
        plt.imshow(depths[0])
        plt.show()
        plt.imshow(depths[0])
        plt.show()
        """

def test_test_data_samlping():
    rmrc_data_path = "/home/sumit/Documents/repo/datasets/rmrc/test"
    test = Rmrc(data_path = rmrc_data_path, training_data = False)
    for i in range(3):
        images = test.next_random_batch(batch_size = 100)
        print(np.shape(images))
        images = images[0].transpose(1, 2, 0)
        plt.imshow(images)
        plt.show()


def test_work_file():
    file = open("workfile", "w")
    for i in range(10):
        print(i, file = file)

if __name__ == "__main__":
    #test_work_file()
    #load_rmrc()
    #augment_data()
    #test_data_loader()
    test_data_sampling()
    #test_test_data_samlping()
