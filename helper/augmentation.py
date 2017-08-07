import numpy as np
import torch

def rotate180(image):
    img_rot180 = np.rot90(image, 2)
    return img_rot180

def inverse_rotate180(rotated_image):
    img_rot180 = np.rot90(rotated_image, 2)
    return img_rot180

# reflection wrt y-axis
def fliplr(image):
    return np.fliplr(image)

# def inverse_fliplr(flipped_image):
#     return np.fliplr(flipped_image)

def get_inverse_fliplr(cuda = False):
    if cuda==True:
        return inverse_fliplr_cuda
    else:
        return inverse_fliplr

def inverse_fliplr(flipped_image):
    #assume the input to be a tensor and the depth image
    #For lr we must reverse the column order i.e position 2

    col_size = int(flipped_image.size()[1])

    index = np.flip(np.arange(col_size), 0)
    indexTensor = torch.LongTensor(index.copy())

    flippedTranspose = flipped_image.transpose(0, 1)
    original_image_t = flippedTranspose[indexTensor]
    original_image = original_image_t.transpose(0, 1)

    return original_image

def inverse_fliplr_cuda(flipped_image):
    #assume the input to be a tensor and the depth image
    #For lr we must reverse the column order i.e position 2

    col_size = int(flipped_image.size()[1])

    index = np.flip(np.arange(col_size), 0)
    indexTensor = torch.LongTensor(index.copy()).cuda(0)

    flippedTranspose = flipped_image.transpose(0, 1)
    original_image_t = flippedTranspose[indexTensor]
    original_image = original_image_t.transpose(0, 1)

    return original_image

# reflection wrt x-axis
def flipud(image):
    img_flipud = np.flipud(image)
    return img_flipud

# def inverse_flipud(flipped_image):
#     img_flipud = np.flipud(flipped_image)
#     return img_flipud

def get_inverse_flipud(cuda = False):
    if cuda==True:
        return inverse_flipud_cuda
    else:
        return inverse_flipud


def inverse_flipud(flipped_image):
    # assume the input to be a tensor and the depth image
    # For lr we must reverse the column order i.e position 2
    row_size = int(flipped_image.size()[0])

    index = np.flip(np.arange(row_size), 0)
    indexTensor = torch.LongTensor(index.copy())

    original_image = flipped_image[indexTensor]

    return original_image

def inverse_flipud_cuda(flipped_image):
    # assume the input to be a tensor and the depth image
    # For lr we must reverse the column order i.e position 2
    row_size = int(flipped_image.size()[0])

    index = np.flip(np.arange(row_size), 0)
    indexTensor = torch.LongTensor(index.copy()).cuda(0)

    original_image = flipped_image[indexTensor]

    return original_image

# bitwise inverse
def invert_colors(image):
    if(len(image.shape) == 3):
        img_invert = np.invert(image)
    else:
        img_invert = image
    return img_invert

def return_same_image(image):
    return image
