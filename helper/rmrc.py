import os
import numpy as np
import pickle
import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
#import cv2

class rmrcData(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        crop_size = (232, 310)
        scale = 240
        scale_2 = (55, 75)

        self.to_tensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(crop_size)
        self.to_PIL = transforms.ToPILImage()
        self.scale = transforms.Scale(scale)
        #self.scale_2 = transforms.Scale(scale_2)
    def __getitem__(self, index):
        
        img = self.X[index]
        target = self.y[index]
        
        #img = transforms.Compose([transforms.ToPILImage(), transforms.Scale(240),transforms.CenterCrop(self.crop_size), transforms.ToTensor()])(img.transpose(1,2,0))
        img = self.to_PIL(img.transpose(1,2,0))
        #img = self.scale(img)
        #img = self.center_crop(img)
        img = self.to_tensor(img)
        #img = torch.from_numpy(img)

        #target = self.to_PIL(target.transpose(2,1,0))
        #target = self.scale(target)
        #target = self.center_crop(target)
        #target = self.scale_2(target)
        # to fit scale 2 outputs
        #target = target.resize((75, 55), Image.BILINEAR)
        #size_1 = target.size[1]
        #size_0 = target.size[0]
        #target = torch.FloatTensor(target.getdata())
        #target = target.view(1, size_1, size_0)
        target = torch.from_numpy(target)
      
        return img, target

    def __len__(self):
        return len(self.y)


class Rmrc(object):

    def __init__(self, data_path=None, training_data = True):
        self.training_data = training_data
        # path to data directory
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            print("Are you high!! Data path specified does not even exist!")

        # list of data files
        self.data_files = os.listdir(data_path)
        self.data_files.sort()

        # current data file being processed
        self.current_file_num = 0

        # data object
        self.rmrc_data = None

        # number of samples in a data file
        self.num_samples = -1

        # number of batches sampled from a data file
        self.num_batches_sampled = 0

        # max number of batch sampling allowed froma a file
        self.max_sampling_from_file = 0

        self.start = 0
        self.end = 0

    def read_rmrc_data(self, data_path):
        with open(data_path, "rb") as file:
            rmrc_data = pickle.load(file)
        return rmrc_data

    def next_random_batch(self, batch_size):
        if self.rmrc_data is None:
            print("current file number : ", self.current_file_num)
            data_file_path = os.path.join(self.data_path, self.data_files[self.current_file_num])
            self.rmrc_data = self.read_rmrc_data(data_file_path)
            self.num_samples, _, _, _ = np.shape(self.rmrc_data['images'])
            self.max_sampling_from_file = self.num_samples / batch_size + 10
            self.num_batches_sampled = 0

        mask = np.random.choice(range(self.num_samples), batch_size)
        self.num_batches_sampled += 1
        images = self.rmrc_data['images'][mask]
        #images = cv2.normalize(images, images, 0, 1, cv2.NORM_MINMAX)
        images = images/255
        depths = None
        if self.training_data:
            depths = self.rmrc_data['depths'][mask]

        if self.num_batches_sampled >= self.max_sampling_from_file:
            if self.current_file_num < len(self.data_files) - 1:
                self.current_file_num += 1
            else:
                self.current_file_num = 0
            self.rmrc_data = None
        if self.training_data:
            return (images, depths)
        else:
            return images

    def next_batch(self, batch_size):
        if self.rmrc_data is None:
            print("current file number : ", self.current_file_num)
            data_file_path = os.path.join(self.data_path, self.data_files[self.current_file_num])
            self.rmrc_data = self.read_rmrc_data(data_file_path)
            self.num_samples, _, _, _ = np.shape(self.rmrc_data['images'])
            self.max_sampling_from_file = int(self.num_samples / batch_size)
            self.num_batches_sampled = 0
            self.start = 0
            self.end = batch_size

        mask = range(self.start, self.end)
        self.start = self.end
        if self.num_samples - self.end >= batch_size:
            self.end = self.end + batch_size
        else:
            self.end = self.num_samples
        #self.end = self.end + batch_size if self.num_samples - self.end >= batch_size else self.num_samples
        self.num_batches_sampled += 1
        images = self.rmrc_data['images'][mask]
        #images = images / 255
        depths = None
        if self.training_data:
            depths = self.rmrc_data['depths'][mask]

        if self.num_batches_sampled >= self.max_sampling_from_file:
            if self.current_file_num < len(self.data_files) - 1:
                self.current_file_num += 1
            else:
                self.current_file_num = 0
            #self.current_file_num += 1 if self.current_file_num < len(self.data_files) - 1 else 0
            self.rmrc_data = None

        if self.training_data:
            return (images, depths)
        else:
            return images

    def first(self):
        data_file_path = os.path.join(self.data_path, self.data_files[self.current_file_num])
        self.rmrc_data = self.read_rmrc_data(data_file_path)
        images = self.rmrc_data['images'][[0]]
        depths = self.rmrc_data['depths'][[0]]
        return (images, depths)

    def dim(self):
        #_, depth = self.first()
        #return (depth[0].shape[0], depth[0].shape[1])
        # print('shape: ', depth[0].shape)
        #return (232, 310)
        return (92, 120)
        #return(146, 196)