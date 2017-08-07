import os
import h5py
import pickle
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import zoom
import cv2

# comment this path and create a new vaiable according to your own path.
rmrc_data_path = "/home/sumit/Documents/repo/datasets/rmrc/train"

def mat_to_binary_rgbd():
    print("generating binary files...")
    files = os.listdir(rmrc_data_path)
    for itr, file in enumerate(files):
        file_path = os.path.join(rmrc_data_path, file)
        print(file_path)
        if os.path.isfile(file_path) and ".mat" in file:
            file_path = os.path.join(rmrc_data_path, file)
            rmrc_data = h5py.File(file_path)
            data = {}
            binary_data_file_name = file.split('.')[0]
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file = open(binary_file_path, "wb")
            for k, v in rmrc_data.items():
                new_data = []
                if k == 'images':
                    numpy_v = np.array(v)
                    N, _, _, _ = np.shape(numpy_v)
                    for i in range(N):
                        old_image = numpy_v[i]
                        old_image = old_image.transpose(1, 2, 0)
                        # (W,H) = (320,240)
                        new_image = cv2.resize(old_image, (206, 154))
                        # after center cropping (232,310)
                        new_image = new_image[4:150, 5:201]
                        new_image = new_image.transpose(2, 0, 1)
                        new_data.append(new_image)
                    data[k] = np.array(new_data)
                    print(np.shape(data[k]))
                elif k == 'depths':
                    numpy_v = np.array(v)
                    new_data = []
                    N, _, _, = np.shape(numpy_v)
                    for i in range(N):
                        old_image = numpy_v[i]
                        new_image = cv2.resize(old_image, (206, 154))
                        # after center cropping (232,310)
                        new_image = new_image[4:150, 5:201]
                        new_data.append(new_image)
                    data[k] = np.array(new_data)
                    print(np.shape(data[k]))
            pickle.dump(data, binary_file)
    print('data successfully converted..')

def mat_to_binary_rgbd_segmentation():
    print("generating binary files...")
    files = os.listdir(rmrc_data_path)
    for itr, file in enumerate(files):
        file_path = os.path.join(rmrc_data_path, file)
        print(file_path)
        if os.path.isfile(file_path) and ".mat" in file:
            file_path = os.path.join(rmrc_data_path, file)
            rmrc_data = h5py.File(file_path)
            binary_data_file_name = file.split('.')[0]

            data1 = {}
            binary_data_file_name = binary_data_file_name + str(1)
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file1 = open(binary_file_path, "wb")

            data2 = {}
            binary_data_file_name = binary_data_file_name + str(2)
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file2 = open(binary_file_path, "wb")

            data3 = {}
            binary_data_file_name = binary_data_file_name + str(3)
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file3 = open(binary_file_path, "wb")

            data4 = {}
            binary_data_file_name = binary_data_file_name + str(4)
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file4 = open(binary_file_path, "wb")

            data5 = {}
            binary_data_file_name = binary_data_file_name + str(5)
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file5 = open(binary_file_path, "wb")

            for k, v in rmrc_data.items():
                if k == 'images':
                    images = np.array(v)
                    print(np.shape(images))
                    img_set1 = images[:300,:,:,:]
                    data1['images'] = img_set1
                    img_set2 = images[300:600, :, :, :]
                    data2['images'] = img_set2
                    img_set3 = images[600:900, :, :, :]
                    data3['images'] = img_set3
                    img_set4 = images[900:1200, :, :, :]
                    data4['images'] = img_set4
                    img_set5 = images[1200:1500, :, :, :]
                    data5['images'] = img_set5
                if k == 'depths':
                    depths = np.array(v)
                    print(np.shape(depths))
                    depth_set1 = depths[:300, :, :]
                    data1['depths'] = depth_set1
                    depth_set2 = depths[300:600, :, :]
                    data2['depths'] = depth_set2
                    depth_set3 = depths[600:900, :, :]
                    data3['depths'] = depth_set3
                    depth_set4 = depths[900:1200, :, :]
                    data4['depths'] = depth_set4
                    depth_set5 = depths[1200:1500, :, :]
                    data5['depths'] = depth_set5
                if k == 'labels':
                    labels = np.array(v)
                    print(np.shape(labels))
                    label_set1 = labels[:300, :, :]
                    data1['labels'] = label_set1
                    label_set2 = labels[300:600, :, :]
                    data2['labels'] = label_set2
                    label_set3 = labels[600:900, :, :]
                    data3['labels'] = label_set3
                    label_set4 = labels[900:1200, :, :]
                    data4['labels'] = label_set4
                    label_set5 = labels[1200:1500, :, :]
                    data5['labels'] = label_set5
            pickle.dump(data1, binary_file1)
            pickle.dump(data2, binary_file2)
            pickle.dump(data3, binary_file3)
            pickle.dump(data4, binary_file4)
            pickle.dump(data5, binary_file5)
    print('data successfully converted..')


def create_augmented_data():
    print("generating binary files...")
    files = os.listdir(rmrc_data_path)
    for itr, file in enumerate(files):
        file_path = os.path.join(rmrc_data_path, file)
        if os.path.isfile(file_path) and ".mat" not in file:
            file_path = os.path.join(rmrc_data_path, file)
            print(file_path)
            augmented_data = {}
            augmented_images = []
            augmented_depths = []
            with open(file_path, "rb") as read_file:
                data = pickle.load(read_file)
                N, _, _, _ = np.shape(data['images'])
                images = data['images']
                depths = data['depths']
                for i in range(N):
                    print(i)
                    img = images[i]
                    depth = depths[i]
                    augmented_images.append(img)
                    augmented_depths.append(depth)

                    img_rot180 = np.rot90(img, 2)
                    depth_rot180 = np.rot90(depth, 2)
                    augmented_images.append(img_rot180)
                    augmented_depths.append(depth_rot180)

                    img_fliplr = np.fliplr(img)
                    depth_fliplr = np.fliplr(depth)
                    augmented_images.append(img_fliplr)
                    augmented_depths.append(depth_fliplr)

                    img_flipud = np.flipud(img)
                    depth_flipud = np.flipud(depth)
                    augmented_images.append(img_flipud)
                    augmented_depths.append(depth_flipud)

                    img_invert = np.invert(img)
                    depth_invert = depth
                    augmented_images.append(img_invert)
                    augmented_depths.append(depth_invert)

            augmented_data['images'] = np.array(augmented_images)
            augmented_data['depths'] = np.array(augmented_depths)
            binary_data_file_name = file.split('.')[0]
            binary_file_path = os.path.join(rmrc_data_path, binary_data_file_name)
            binary_file = open(binary_file_path, "wb")
            pickle.dump(augmented_data, binary_file)
    print('data successfully converted..')


if __name__ == "__main__":
    mat_to_binary_rgbd()
    #mat_to_binary_rgbd_segmentation()
    #create_augmented_data()