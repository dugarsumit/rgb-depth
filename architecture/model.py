import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import numpy as np
from torch.autograd import Variable


def get_output_size_conv_maxpool(input_size, kernel_size, stride=1, padding=0, dilation=1):
    # Gets the output_size for Conv2d and maxpool layers according to the following equation:
    # `H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    input_height = input_size[0]
    input_width = input_size[1]
    output_width = np.floor((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    output_height = np.floor((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return (output_height, output_width)


def get_stride_size_conv_maxpool(input_size, output_size, kernel_size, padding=0, dilation=1):
    # Gets the output_size for Conv2d and maxpool layers according to the following equation:
    # `H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0]
    if output_size == 1:
        stride_size = 1
    else:
        stride_size = np.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / (output_size - 1))
    return stride_size


def calc_scale_1_size(input_height, input_width):
    h = input_height
    w = input_width
    # All conv layers have normal zero padding/have no effect on size
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 2)
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 2)
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 2)
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 2)
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 2)
    return (int(h), int(w))


def calc_scale_2_size(input_height, input_width):
    # Only layer 2.1 changes the input size
    h = input_height
    w = input_width
    (h, w) = get_output_size_conv_maxpool((h, w), 9, 2, 0)
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 2)
    return (int(h), int(w))


def calc_scale_3_size(input_height, input_width):
    # Only layer 3.1 changes the input size
    h = input_height
    w = input_width
    (h, w) = get_output_size_conv_maxpool((h, w), 9, 2, 0)
    (h, w) = get_output_size_conv_maxpool((h, w), 2, 1)
    return (int(h), int(w))


class Model(nn.Module):
    def __init__(self, input_dim, pretrained_net='vgg16', D=1, train_upsampler=True, save_model=False, cuda=False):

        def set_sizes():
            self.input_height = input_dim[0]
            self.input_width = input_dim[1]
            self.scale_1_size = calc_scale_1_size(self.input_height, self.input_width)
            self.scale_2_size = calc_scale_2_size(self.input_height, self.input_width)
            self.scale_3_size = calc_scale_3_size(self.input_height, self.input_width)
            # print(self.scale_1_size, self.scale_2_size, self.scale_3_size)

        def scale_1():
            if (pretrained_net == 'vgg16'):
                vgg16 = models.vgg16(pretrained=True)
                layers = nn.Sequential()
                skip_1_1_output = nn.Sequential()
                skip_1_2_output = nn.Sequential()
                count = 0
                # Extract all vgg_layers to get the skip layers
                for name, module in vgg16.features.named_children():
                    layers.add_module(name=name, module=module)
                    if (isinstance(module, nn.MaxPool2d)):
                        count += 1
                        if (count == 3):
                            # self.scale_1['layers_upto_1_3'] = layers
                            self.scale_1_section_1 = layers
                            # skip_1_1_output.add_module(name = 'sc_1_layers_upto_1_3', module = layers)
                            skip_1_1_output.add_module(name='sc_1_skip_layer_1_op',
                                                       module=torch.nn.Conv2d(256, 64, 5, padding=2))
                            skip_1_1_output.add_module(name='sc_1_skip_1_upsampled',
                                                       module=torch.nn.UpsamplingBilinear2d(
                                                           size=(self.scale_2_size[0], self.scale_2_size[1])))
                            layers = nn.Sequential()
                        elif (count == 4):
                            self.scale_1_section_2 = layers
                            # skip_1_2_output.add_module(name='sc_1_layers_frm_1_3_to_1_4', module=layers)
                            skip_1_2_output.add_module(name='sc_1_skip_layer_2_op',
                                                       module=torch.nn.Conv2d(512, 64, 5, padding=2))
                            skip_1_2_output.add_module(name='sc_1_skip_2_upsampled',
                                                       module=torch.nn.UpsamplingBilinear2d(
                                                           size=(self.scale_2_size[0], self.scale_2_size[1])))
                            layers = nn.Sequential()
                self.scale_1_section_3 = layers
                self.scale_1_skip_1_1_output = skip_1_1_output
                self.scale_1_skip_1_2_output = skip_1_2_output

                # 2 fully connected layers
                layers = nn.Sequential()
                layers.add_module(name='sc_1_dropout_1', module=nn.Dropout(p=0.5))
                layers.add_module(name='sc_1_fc_layer_1',
                                  module=nn.Linear(in_features=self.scale_1_size[0] * self.scale_1_size[1] * 512,
                                                   out_features=self.scale_2_size[0] * self.scale_2_size[1] * D,
                                                   bias=True))
                layers.add_module(name='sc_1_dropout_2', module=nn.Dropout(p=0.5))
                layers.add_module(name='sc_1_fc_layer_2',
                                  module=nn.Linear(in_features=self.scale_2_size[0] * self.scale_2_size[1] * D,
                                                   out_features=self.scale_2_size[0] * self.scale_2_size[1] * D,
                                                   bias=True))
                self.scale_1_section_4 = layers

            else:
                print('Initialization failed, supply the correct parameter {''vgg16''}')
                return

        def scale_2():
            self.scale_2_section_1 = nn.Sequential(
                # 2.1
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=9, stride=2, padding=0, bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
            self.scale_2_section_2 = nn.Sequential(
                # 2.2
                nn.Conv2d(in_channels=96 + 64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU(),
                # 2.3
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU()
            )
            self.scale_2_section_3 = nn.Sequential(
                # 2.4
                nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU(),
                # 2.5
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU()
            )
            self.scale_2_section_4 = nn.Sequential(
                # 2.6
                nn.Conv2d(in_channels=64 + D, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU(),
                # 2.7
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU(),
                # 2.8
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU()
            )

            self.scale_2_section_5 = nn.Sequential(
                # 2.9
                nn.Conv2d(in_channels=64, out_channels=D, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU(),
            )
            self.scale_2_section_6 = nn.Sequential(
                # 2.9
                nn.UpsamplingBilinear2d(size=(self.scale_3_size[0], self.scale_3_size[1]))
            )
            self.scale_2_section_7 = nn.Sequential(
                # Upsampling
                nn.UpsamplingBilinear2d(size=(self.input_height, self.input_width))
            )



        super(Model, self).__init__()
        self.D = D
        self.flag = train_upsampler
        self.save_model = save_model
        # scale 1
        set_sizes()
        scale_1()
        # print("scale_1 defined")
        # scale 2
        scale_2()
        # print("scale_2 defined")

        if cuda == True:
            self.scale_1_section_1 = self.scale_1_section_1.cuda(0)  # 1.1, 1.2, 1.3
            self.scale_1_skip_1_1_output = self.scale_1_skip_1_1_output.cuda(0)  # skip 1.1
            self.scale_1_section_2 = self.scale_1_section_2.cuda(0)  # 1.4
            self.scale_1_skip_1_2_output = self.scale_1_skip_1_2_output.cuda(0)  # skip 1.2
            self.scale_1_section_3 = self.scale_1_section_3.cuda(0)  # 1.5
            self.scale_1_section_4 = self.scale_1_section_4.cuda(0)  # 1.6

            self.scale_2_section_1 = self.scale_2_section_1.cuda(0)  # 2.1
            self.scale_2_section_2 = self.scale_2_section_2.cuda(0)  # 2.2, 2.3
            self.scale_2_section_3 = self.scale_2_section_3.cuda(0)  # 2.4, 2.5
            self.scale_2_section_4 = self.scale_2_section_4.cuda(0)  # 2.6, 2.7, 2.8
            self.scale_2_section_5 = self.scale_2_section_5.cuda(0)  # 2.9
            self.scale_2_section_6 = self.scale_2_section_6.cuda(0)
            self.scale_2_section_7 = self.scale_2_section_7.cuda(0)

        # Generally this code is not used, but can be used if we want to train all three scales once
        if (train_upsampler == True):
            self.scale_3()
            if cuda == True:
                self.scale_3_section_1 = self.scale_3_section_1.cuda(0)
                self.scale_3_section_2 = self.scale_3_section_2.cuda(0)
                self.scale_3_section_3 = self.scale_3_section_3.cuda(0)
                self.scale_3_section_4 = self.scale_3_section_4.cuda(0)
                self.scale_3_section_5 = self.scale_3_section_5.cuda(0)

    def scale_3(self):

        self.scale_3_section_1 = nn.Sequential(
            # 3.1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=9, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )
        self.scale_3_section_2 = nn.Sequential(
            # 3.2
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU()
        )
        self.scale_3_section_3 = nn.Sequential(
            # 3.3
            nn.Conv2d(in_channels=64 + self.D, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 3.4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            # 3.5
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU()
        )
        self.scale_3_section_4 = nn.Sequential(
            # 3.6
            nn.Conv2d(in_channels=64, out_channels=self.D, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU()
        )
        self.scale_3_section_5 = nn.Sequential(
            # Upsampling
            nn.UpsamplingBilinear2d(size=(self.input_height, self.input_width))
        )

    def add_upsampler_2_model(self, cuda=False):
        self.scale_3()
        if cuda == True:
            self.scale_3_section_1 = self.scale_3_section_1.cuda(0)
            self.scale_3_section_2 = self.scale_3_section_2.cuda(0)
            self.scale_3_section_3 = self.scale_3_section_3.cuda(0)
            self.scale_3_section_4 = self.scale_3_section_4.cuda(0)
            self.scale_3_section_5 = self.scale_3_section_5.cuda(0)

    def forward(self, x):
        # scale_1
        # TODO:: Rename x to sc_1_main_op, sc_2_main_op, sc_3_main_op and the skip layers as such.

        sc_1_main_op = self.scale_1_section_1(x)  # 1.1, 1.2, 1.3
        sc_1_skip_1_1 = self.scale_1_skip_1_1_output(sc_1_main_op)  # skip 1.1
        sc_1_main_op = self.scale_1_section_2(sc_1_main_op)  # 1.4
        sc_1_skip_1_2 = self.scale_1_skip_1_2_output(sc_1_main_op)  # skip 1.2
        sc_1_main_op = self.scale_1_section_3(sc_1_main_op)  # 1.5
        sc_1_main_op = sc_1_main_op.view(sc_1_main_op.size(0), -1)
        sc_1_main_op = self.scale_1_section_4(sc_1_main_op)  # 1.6

        sc_1_main_op = sc_1_main_op.view(-1, 1, self.scale_2_size[0], self.scale_2_size[1])  # 1.7 reshape
        # layer = nn.UpsamplingBilinear2d(size = (self.input_height, self.input_width))
        # sc_1_main_op = layer(sc_1_main_op)
        # print("scale_1: ", sc_1_main_op.size())
        # return torch.squeeze(sc_1_main_op)

        # scale_2
        sc_2_main_op = self.scale_2_section_1(x)  # 2.1
        sc_2_main_op = self.scale_2_section_2(torch.cat((sc_2_main_op, sc_1_skip_1_1), dim=1))  # 2.2, 2.3
        sc_2_main_op = self.scale_2_section_3(torch.cat((sc_2_main_op, sc_1_skip_1_2), dim=1))  # 2.4, 2.5
        sc_2_main_op = self.scale_2_section_4(torch.cat((sc_2_main_op, sc_1_main_op), dim=1))  # 2.6, 2.7, 2.8
        sc_2_main_op = self.scale_2_section_5(sc_2_main_op)  # 2.9
        upsampled_2x_sc2_main_op = self.scale_2_section_6(sc_2_main_op)
        upsampled_4x_sc2_main_final = self.scale_2_section_7(
            sc_2_main_op)  # Not part of model. Upsampled to original image size
        # print("scale_2: ", sc_2_main_op.size(), upsampled_4x_sc2_main_final.size())


        if (self.flag == False):
            return torch.squeeze(upsampled_4x_sc2_main_final)

        # scale_3
        sc_3_main_op = self.scale_3_section_1(x)  # 3.1
        sc_3_main_op = self.scale_3_section_2(sc_3_main_op)  # 3.2
        sc_3_main_op = self.scale_3_section_3(torch.cat((sc_3_main_op, upsampled_2x_sc2_main_op), dim=1))  # 3.3, 3.4, 3.5
        sc_3_main_op = self.scale_3_section_4(sc_3_main_op)  # 3.6
        upsampled_sc_3_main_op = self.scale_3_section_5(sc_3_main_op)  # Not part of model. Upsampled to original image size
        # print("scale_3: ", sc_3_main_op.size(), upsampled_sc_3_main_op.size())

        return torch.squeeze(upsampled_sc_3_main_op)


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        if self.save_model:
            print('Saving model... %s' % path)
            torch.save(self, path)


# testing
if __name__ == '__main__':
    print('model.py')

    # test_model = Model().cuda(device_id=0)

    # Simple forward check

    # x = torch.rand(2, 3, 232, 310)
    # test_model = Model(input_dim=(232, 310), train_upsampler=True)
    # y = test_model.forward(Variable(x))
    # print(y.size())

    # prints the model structure: NOTE: Whenever you make any changes to this file, please print the model structure
    # and check to see if it looks okay to you.
    # for name, module in test_model.named_children():
    #    print (name, module)
