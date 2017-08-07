import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
from helper.visualizer import *
from helper.metrics import *
from helper import augmentation
import math
import os


class Trainer(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    
    def __init__(self, model, optim = torch.optim.Adam, loss_func = torch.nn.MSELoss(), args_adam = {},
                augmentation_list = ['rotate180', 'fliplr', 'flipud', 'invert_colors'], reg = 1, train_upsampler = False, cuda=False, accuracy_thresholds = 1.5, layer_hyper_params=None):
        self.model = model
        args_update = self.default_adam_args.copy()
        args_update.update(args_adam)
        self.optim = optim
        self.loss_func = loss_func
        self.optim_args = args_update
        self.train_upsampler = train_upsampler
        self.accuracy_thresholds = accuracy_thresholds
        weight_decay = 0.0
        default_layer_hyper_params = {'lr': {'scale_1_section_1': 1e-4,
                                     'scale_1_section_2': 1e-4,
                                     'scale_1_section_3': 1e-4,
                                     'scale_1_section_4': 1e-1,
                                     'scale_1_skip_1_1_output': 1e-2,
                                     'scale_1_skip_1_2_output': 1e-2,
                                     'scale_2_section_1': 1e-3,
                                     'scale_2_section_2': 1e-2,
                                     'scale_2_section_3': 1e-2,
                                     'scale_2_section_4': 1e-2,
                                     'scale_2_section_5': 1e-3,
                                     'scale_3_section_1': 1e-3,
                                     'scale_3_section_2': 1e-2,
                                     'scale_3_section_3': 1e-2,
                                     'scale_3_section_4': 1e-3,
                                     },
                              'weight_decay': 0
                              }
        for keyname in layer_hyper_params.keys():
            default_layer_hyper_params[keyname] = layer_hyper_params[keyname]

        layer_hyper_params = default_layer_hyper_params


        if self.train_upsampler:
            self.freeze_network()
            self.optim_list = [{'params': model.scale_3_section_1.parameters(), 'lr': layer_hyper_params['lr']['scale_3_section_1'], 'weight_decay': layer_hyper_params['weight_decay']},
                                {'params': model.scale_3_section_2.parameters(), 'lr': layer_hyper_params['lr']['scale_3_section_2'], 'weight_decay': layer_hyper_params['weight_decay']},
                                {'params': model.scale_3_section_3.parameters(), 'lr': layer_hyper_params['lr']['scale_3_section_3'], 'weight_decay': layer_hyper_params['weight_decay']},
                                {'params': model.scale_3_section_4.parameters(), 'lr': layer_hyper_params['lr']['scale_3_section_4'], 'weight_decay': layer_hyper_params['weight_decay']}]
        else:
            self.optim_list = [{'params':model.scale_1_section_1.parameters(),'lr': layer_hyper_params['lr']['scale_1_section_1'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_1_section_2.parameters(),'lr': layer_hyper_params['lr']['scale_1_section_2'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_1_section_3.parameters(),'lr': layer_hyper_params['lr']['scale_1_section_3'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_1_section_4.parameters(),'lr': layer_hyper_params['lr']['scale_1_section_4'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_1_skip_1_1_output.parameters(),'lr': layer_hyper_params['lr']['scale_1_skip_1_1_output'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_1_skip_1_2_output.parameters(),'lr': layer_hyper_params['lr']['scale_1_skip_1_2_output'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_2_section_1.parameters(),'lr': layer_hyper_params['lr']['scale_2_section_1'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_2_section_2.parameters(),'lr': layer_hyper_params['lr']['scale_2_section_2'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_2_section_3.parameters(),'lr': layer_hyper_params['lr']['scale_2_section_3'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_2_section_4.parameters(),'lr': layer_hyper_params['lr']['scale_2_section_4'], 'weight_decay': layer_hyper_params['weight_decay']},
                           {'params':model.scale_2_section_5.parameters(),'lr': layer_hyper_params['lr']['scale_2_section_5'], 'weight_decay': layer_hyper_params['weight_decay']}]

        self.reset_histories()
        self.initialize_augmentation(augmentation_list = augmentation_list, cuda=cuda)
        self.reg = reg

    def freeze_network(self):
        model = self.model
        model_layers = [model.scale_1_section_1, model.scale_1_section_2, model.scale_1_section_3,
                        model.scale_1_section_4, model.scale_1_skip_1_1_output, model.scale_1_skip_1_2_output,
                        model.scale_2_section_1, model.scale_2_section_2, model.scale_2_section_3,
                        model.scale_2_section_4, model.scale_2_section_5]
        for layer in model_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def initialize_augmentation(self, augmentation_list, cuda=False):
        aug_func = []
        inverse_aug_func = []
        # For Original Image
        aug_func.append(augmentation.return_same_image)
        inverse_aug_func.append(augmentation.return_same_image)
        for i in range(len(augmentation_list)):
            if(augmentation_list[i] == 'rotate180'):
                #print("rotate")
                aug_func.append(augmentation.rotate180)
                inverse_aug_func.append(augmentation.inverse_rotate180)
            elif(augmentation_list[i] == 'fliplr'):
                #print("fliplr")
                aug_func.append(augmentation.fliplr)
                inverse_aug_func.append(augmentation.get_inverse_fliplr(cuda=cuda))
            elif (augmentation_list[i] == 'flipud'):
                #print("flipud")
                aug_func.append(augmentation.flipud)
                inverse_aug_func.append(augmentation.get_inverse_flipud(cuda=cuda))
            elif (augmentation_list[i] == 'invert_colors'):
                #print("invert_colors")
                aug_func.append(augmentation.invert_colors)
                inverse_aug_func.append(augmentation.return_same_image)
        self.augmentation_func = aug_func
        self.inverse_augmentation_func = inverse_aug_func

    def reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def train(self, result_file, train_loader, val_loader, save_model_path, loss_graph_path, acc_graph_path, result_img_path = None, num_epochs = 10,
              nth = 1, lr_decay = 0.1, cuda = True, batch_size=1, num_iterations=10, use_set_loss = False, overfit = False,
              randomize_batch = False):
        self.reset_histories()
        model = self.model
        optim = self.optim(self.optim_list, **self.optim_args)
        optim.zero_grad()

        total_it = num_iterations * num_epochs
        counter = 0
        print("training started...")
        for epoch in range(num_epochs):
            train_scores = []
            #lr = ((epoch+1) % 10 == 0 and lr * lr_decay or lr)
            #optim.param_groups[0]['lr'] = lr
            #print 'lr: ' + str(optim.state_dict()['param_groups'][0]['lr'])
            for itr in range(num_iterations):
                print(itr)
                if overfit:
                    xbatch, ybatch = train_loader.first()
                else:
                    if randomize_batch:
                        xbatch, ybatch = train_loader.next_random_batch(batch_size = batch_size)
                    else:
                        xbatch, ybatch = train_loader.next_batch(batch_size = batch_size)
                #ybatch = np.squeeze(ybatch)
                #ybatch[np.isnan(ybatch)] = -1
                xbatch, ybatch = self.preprocess(xbatch, ybatch)
                #xbatch = torch.FloatTensor(np.array(xbatch, dtype = float))
                #ybatch = torch.FloatTensor(np.array(ybatch, dtype = float))
                # fetch augmented tensor
                if cuda:
                    x = Variable(xbatch.cuda(0))
                    y = Variable(ybatch.cuda(0))
                else:
                    x = Variable(xbatch)
                    y = Variable(ybatch)
                #print(x)
                predicted_depth = model.forward(x)
                #print(predicted_depth)
                #print(np.shape(predicted_depth.data.numpy()))
                #print(np.shape(y.data.numpy()))

                #print("done with the fwd pass...")
                loss = self.loss_func(predicted_depth, y, self.reg, self.inverse_augmentation_func, use_set_loss)

                if (itr >= num_iterations-1):
                    # print('zero vals pred: ', np.sum(predicted_depth.cpu().data.numpy() == 0), ', zero vals actual: ',
                    #       np.sum(y.cpu().data.numpy() == 0), ', loss: ', loss.type(torch.FloatTensor).data.numpy())
                    # print('abs diff', np.sum(np.absolute(
                    #     predicted_depth.cpu().data.numpy()[np.isfinite(y.cpu().data.numpy())] - y.cpu().data.numpy()[
                    #         np.isfinite(y.cpu().data.numpy())])))
                    saveimgpath = None
                    saveImg = False
                    if(result_img_path != None):
                        saveimgpath = os.path.join(result_img_path, "img_epoch_" + str(epoch) + "_itr_" + str(itr))
                        saveImg = True
                    show_generated_and_actual_depth(predicted_depth.cpu().data.numpy(), y.cpu().data.numpy(), save=saveImg,
                                                    file=saveimgpath)

                #print(loss)
                #print("done with calculating loss...")
                self.train_loss_history.append(loss.type(torch.FloatTensor).data.numpy())
                train_scores.append(calculate_pixelwise_accuracy_saadhana(predicted_depth, y, self.accuracy_thresholds))
                optim.zero_grad()
                loss.backward()

                # print("done with back prop")
                # self.print_weights(model)
                optim.step()
                # print("------")
                # self.print_weights(model)
                # print("done with updation")

                counter += 1
                if(counter % nth == 0):
                    print_current_loss(itr = counter, total_itr = total_it, loss_history = self.train_loss_history,
                                       result_file = result_file)

            # Compute training accuracy
            self.train_acc_history.append(np.mean(train_scores))
            print_current_accuracy(epoch = epoch,
                                   num_epochs = num_epochs,
                                   acc_history = self.train_acc_history,
                                   type = 'train', result_file = result_file)

            # Compute validation accuracy
            self.val_acc_history.append(compute_val_accuracy(val_loader, model, self.accuracy_thresholds, cuda=cuda))
            print_current_accuracy(epoch = epoch,
                                   num_epochs = num_epochs,
                                   acc_history = self.val_acc_history,
                                   type = 'val', result_file = result_file)

            if ((np.log(epoch + 1) / np.log(3)) % 1 == 0):
                model.save(path=save_model_path + '_epoch' + str(epoch))

            #scaling learning rate
            for params in optim.param_groups:
                params['lr'] *= 0.1


        model.save(path=save_model_path + '_epoch' + str(num_epochs-1))
        plot_loss(loss_history = self.train_loss_history,
                  path = loss_graph_path)
        plot_accuracy(train_acc_history = self.train_loss_history,
                      val_acc_history = self.val_acc_history,
                      path = acc_graph_path)
        model.save(path = save_model_path)
        print("training complete..")

    def print_weights(self, model):
        # model_layers = [model.scale_1_section_1, model.scale_1_section_2, model.scale_1_section_3,
        #                 model.scale_1_section_4, model.scale_1_skip_1_1_output, model.scale_1_skip_1_2_output,
        #                 model.scale_2_section_1, model.scale_2_section_2, model.scale_2_section_3,
        #                 model.scale_2_section_4, model.scale_2_section_5, model.scale_3_section_1, model.scale_3_section_2,
        #                 model.scale_3_section_3, model.scale_3_section_4
        #                 ]
        model_layers = [model.scale_1_section_4]
        max = 1e-10
        for layer in model_layers:
            print(torch.max(layer[3].weight))

    def preprocess(self, images, depths):
        new_images = []
        new_depths = []
        for i in range(images.shape[0]):
            for j in range(len(self.augmentation_func)):
                new_images.append(self.augmentation_func[j](images[i].transpose(1, 2, 0)).transpose(2, 0, 1))
                new_depths.append(self.augmentation_func[j](depths[i]))
        image_tensors = torch.FloatTensor(np.array(new_images, dtype = float))
        depth_tensors = torch.FloatTensor(np.array(new_depths, dtype = float))
        return image_tensors, depth_tensors


#testing
#if __name__== '__main__':
    #only check if an error exists
    #model = Model()
    #tainer = Trainer(model)