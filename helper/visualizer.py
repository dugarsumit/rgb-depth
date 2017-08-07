import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random as rnd
import os
import numpy as np

from matplotlib.dates import num2date


def print_current_loss(itr, total_itr, loss_history, result_file):
    if result_file is not None:
        print('[Iteration' + str(itr) + '/' + str(total_itr) + ']' + 'TRAIN loss:' + str(loss_history[-1]),
              file = result_file)
    else:
        print('[Iteration' + str(itr) + '/' + str(total_itr) + ']' + 'TRAIN loss:' + str(loss_history[-1]))


def print_current_accuracy(epoch, num_epochs, acc_history, result_file, type = 'train'):
    if result_file is not None:
        if type == 'train':
            print('[Epoch' + str(epoch) + '/' + str(num_epochs) + '] Train   acc:' + str(acc_history[-1]),
                  file = result_file)
        elif type == 'val':
            print('[Epoch' + str(epoch) + '/' + str(num_epochs) + '] Val   acc:' + str(acc_history[-1]),
                  file = result_file)
    else:
        if type == 'train':
            print('[Epoch' + str(epoch) + '/' + str(num_epochs) + '] Train   acc:' + str(acc_history[-1]))
        elif type == 'val':
            print('[Epoch' + str(epoch) + '/' + str(num_epochs) + '] Val   acc:' + str(acc_history[-1]))


def plot_loss(loss_history, path):
    plt.title("Training Loss")
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig(path+".png")
    #print("loss graph saved at ", path)
    # Clear Plots
    plt.gcf().clear()

def plot_loss_from_workfile(points, loss_history, path):
    plt.title("Training Loss")
    plt.plot(points, loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig(path+".png")
    #print("loss graph saved at ", path)
    # Clear Plots
    plt.gcf().clear()

def plot_accuracy(train_acc_history, val_acc_history, path):
    plt.title('Accuracy')
    plt.plot(train_acc_history, '-bo', label = 'train')
    plt.plot(val_acc_history, '-go', label = 'val')
    plt.plot([0.5] * len(val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc = 'lower right')
    plt.gcf().set_size_inches(15, 12)
    #plt.show()
    plt.savefig(path+".png")
    # Clear Plots
    plt.gcf().clear()
    print("acc graph saved at ", path)

def view_result(image, depth):
    image = image.transpose(1, 2, 0)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(depth)
    plt.axis('off')
    plt.show()
    # Clear Plots
    plt.gcf().clear()


def show_generated_and_actual_depth(generated, actual, save=False, file=None):
    print('Generated shape: ', generated.shape)
    print('Actual shape: ', actual.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(generated[0])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(actual[0])
    if(save):
        plt.savefig(file)
    else:
        plt.show()
    # Clear Plots
    plt.gcf().clear()

def show_generated_and_actual_depth_all_aug(generated, actual, rgb):
    #print('Generated shape: ', generated.shape)
    #print('Actual shape: ', actual.shape)
    augmentations = generated.shape[0]
    num_imgs = 3*augmentations
    for i in range(num_imgs):
        #fig = plt.figure(figsize=(5,5))
        #fig.add_subplot(num_imgs, augmentations, i+1)
        plt.subplot(num_imgs, augmentations, i + 1)
        if(i<augmentations):
            plt.imshow(generated[i])
        elif(i>=augmentations and i<2*augmentations):
            plt.imshow(actual[i-augmentations])
        else:
            image = rgb[i - 2*augmentations]/255
            image = image.transpose(1, 2, 0)
            plt.imshow(image)
        #plt.axes([0,0,0.7,0.6])
        plt.axis('off')
    plt.show()
    # Clear Plots
    plt.gcf().clear()

def view_2_images(image1, image2):
    image1 = image1.transpose(1, 2, 0)
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    image2 = image2.transpose(1, 2, 0)
    plt.imshow(image2)
    plt.axis('off')
    plt.show()
    # Clear Plots
    plt.gcf().clear()

def view_2_depths(depth1, depth2):
    plt.subplot(1, 2, 1)
    plt.imshow(depth1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(depth2)
    plt.axis('off')
    plt.show()
    # Clear Plots
    plt.gcf().clear()

def view_3_depths(depth1, depth2, depth3):
    plt.subplot(1, 3, 1)
    plt.imshow(depth1)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(depth2)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(depth3)
    plt.axis('off')
    plt.show()
    # Clear Plots
    plt.gcf().clear()

# expecting image in (C,H,W) and depth in (H,W)
def save_result(image, actual_depth, predicted_depth, file_name):
    image = np.squeeze(image)
    image = image / 255
    actual_depth = np.squeeze(actual_depth)
    image = image.transpose(1, 2, 0)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(actual_depth)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_depth)
    plt.axis('off')
    plt.savefig(file_name+".png", bbox_inches='tight')
    # Clear Plots
    plt.gcf().clear()

def generate_loss_from_workfile(num_epochs, num_iterations, nth, path):
    values = []
    with open(os.path.join(path, 'workfile'), 'r') as file:
        for line in file:
            if "loss" in line :
                loss = line.split(':')
                values.append(loss[1][2:-2])
    loss_graph_path = os.path.join(path, 'loss_graph')
    points = range(0, num_epochs*num_iterations, nth)
    plot_loss_from_workfile(points, values, loss_graph_path)
    #return values

def test_visualizer():
    train_acc_history = [rnd.sample(range(1, 20), 10)]
    val_acc_history = [rnd.sample(range(1, 20), 10)]
    loss_history = rnd.sample(range(1, 100), 50)
    loss_graph_path = "../documents/loss"
    print(loss_history)
    plot_loss(loss_history, loss_graph_path)
    #acc_graph_path = "../documents/acc"
    #plot_accuracy(train_acc_history, val_acc_history, acc_graph_path)


if __name__ == '__main__':
    #test_visualizer()
    generate_loss_from_workfile(num_epochs=5,
                                num_iterations=4000,
                                nth=100,
                                path='/home/sumit/Documents/repo/dlcv_proj/results/rmrc_skip_layers_removed_2017_08_04_13_09')
