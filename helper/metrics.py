import numpy as np
from torch.autograd import Variable
import torch
from helper import visualizer
from helper import augmentation


def our_loss_fun(predicted_depths, actual_depths, reg, inverse_aug_func, use_set_loss):
    loss = torch.nn.MSELoss()
    mask = actual_depths == actual_depths
    single_loss = loss(predicted_depths[mask], actual_depths[mask])
    set_loss = 0
    if use_set_loss:
        set_loss = regularizer(predicted_depths, inverse_aug_func)
    #print("set_loss ", set_loss)
    return single_loss + reg * set_loss

def our_loss_fun_sqrt(predicted_depths, actual_depths, reg, inverse_aug_func, use_set_loss):
    loss = torch.nn.MSELoss()
    mask = actual_depths == actual_depths
    single_loss = torch.sqrt(loss(predicted_depths[mask], actual_depths[mask]))
    set_loss = 0
    if use_set_loss:
        # TODO: set_loss is still using mse, use a new regularizer function or add a parameter to change that
        set_loss = regularizer(predicted_depths, inverse_aug_func)
    #print("set_loss ", set_loss)
    return single_loss + reg * set_loss


'''
Here we want to are trying to compute loss between prediction of
augmented images including the original image
'''
def regularizer(predicted_depths, inverse_aug_func):
    num_of_aug = len(inverse_aug_func)
    total_num_samples = len(predicted_depths)
    reg_loss = 0
    for i in range(int(total_num_samples / num_of_aug)):
        for j in range(num_of_aug):
            for k in range(j + 1, num_of_aug):
                index1 = i * num_of_aug + j
                index2 = i * num_of_aug + k
                #print('index1 & 2 : ', index1, index2)
                if (index1 != index2):
                    depthj = inverse_aug_func[j](predicted_depths[index1])
                    #print(inverse_aug_func[j])
                    #print('depthj size: ', depthj.size())
                    depthk = inverse_aug_func[k](predicted_depths[index2])
                    #print(inverse_aug_func[k])
                    #print('depthj size: ', depthk.size())

                    reg_loss += mse_loss(depthj,depthk)
                    #reg_loss += mse_loss(depthj[mask],depthk[mask])
                    #print('Sum reg loss', reg_loss)
    if num_of_aug > 1:
        return (1 / (num_of_aug - 1)) * reg_loss
    else:
        return 0

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def mse_loss_rel(input, target):
    # TODO: This gives nan. Fix it.
    return torch.sum((1 - input/target)*(1 - input/target)) / input.data.nelement()

def calculate_pixelwise_accuracy(pred_depth, actual_depth):
    mask = actual_depth == actual_depth
    return np.mean((pred_depth[mask] == actual_depth[mask]).data.cpu().numpy())

def calculate_pixelwise_accuracy_saadhana(pred_depth, actual_depth, accuracy_thresholds):
    mask = actual_depth == actual_depth
    #print("percentage of Nans in actual depth: ", np.mean((actual_depth != actual_depth).data.cpu().numpy()))
    pred_depth = pred_depth[mask]
    actual_depth = actual_depth[mask]
    #print("percentage of Zeros in actual depth: ", np.mean((actual_depth == 0).data.cpu().numpy()))
    #print("percentage of Zeros in pred depth: ", np.mean((pred_depth == 0).data.cpu().numpy()))
    pred_depth[pred_depth==0]=1e-8
    actual_depth[actual_depth == 0] = 1e-8
    max_ratio = torch.max(torch.div(pred_depth, actual_depth), torch.div(actual_depth, pred_depth))
    accuracy_1 = torch.mean((max_ratio < accuracy_thresholds[0]).float())
    # **Might need in the future. Accuracy with 3 thresholds presented in paper**
    # accuracy_2 = torch.mean((max_ratio < accuracy_thresholds[1]).float())
    # accuracy_3 = torch.mean((max_ratio < accuracy_thresholds[2]).float())
    return (accuracy_1.cpu().data[0])


def compute_val_accuracy(val_loader, model, accuracy_thresholds, cuda = False ):
    val_scores = []
    for itr in range(300):
        x, y = val_loader.next_batch(batch_size = 1)
        x = torch.FloatTensor(np.array(x, dtype = float))
        y = torch.FloatTensor(np.array(y, dtype = float))
        if cuda:
            inputs = Variable(x, requires_grad = False).cuda(0)
            depths = Variable(y, requires_grad = False).cuda(0)
        else:
            inputs = Variable(x, requires_grad = False)
            depths = Variable(y, requires_grad = False)
        pred_depths = model.forward(inputs)
        val_scores.append(calculate_pixelwise_accuracy_saadhana(pred_depths, depths, accuracy_thresholds))
    return np.mean(val_scores)


if __name__ == "__main__":

    from architecture.model import Model
    from architecture.trainer import Trainer
    from architecture.tester import Tester
    from helper.rmrc import Rmrc
    import os
    import dateutil.tz
    import datetime
    import torch

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M')
    # comment this path and create a new vaiable according to your own path.

    # base_path = "/home/sumit/Documents/repo/datasets/rmrc"
    base_path = "/home/neha/Documents/TUM_Books/projects/dlcv_proj/data/rmrc"
    rmrc_train_data_path = os.path.join(base_path, "train")
    rmrc_val_data_path = os.path.join(base_path, "val")
    rmrc_test_data_path = os.path.join(base_path, "test")

    train_dataset = Rmrc(data_path = rmrc_train_data_path)
    val_dataset = Rmrc(data_path = rmrc_val_data_path)
    test_dataset = Rmrc(data_path = rmrc_test_data_path)

    xbatch, ybatch = train_dataset.next_random_batch(batch_size=1)
    print('Is nan xbatch: ', np.sum(np.isnan(xbatch)))
    print('Is nan ybatch: ', np.sum(np.isnan(ybatch)))
    print('Is neg ybatch: ', np.sum(ybatch == 0))

    ybatch[np.isnan(ybatch)] = 0

    print("X batch shape, y batch shape : ", xbatch.shape, ybatch.shape)

    index = np.flip(np.arange(310), 0)
    indexTensor = torch.LongTensor(index.copy())

    ybatchT = torch.FloatTensor(np.array(ybatch, dtype=float))
    ybatchVar = Variable(ybatchT, requires_grad=False)
    ybatchVartranspose = ybatchVar.transpose(0,2)
    ybatchVarindexed = ybatchVartranspose[indexTensor]

    print('Variable size ybatchVarindexed ', ybatchVarindexed.size())
    ybatchVartranspose = ybatchVarindexed.transpose(0, 2)
    ybatchreverse = ybatchVartranspose.data.numpy()

    #visualizer.view_result(xbatch[0], ybatchreverse[0])
    inv_depth = augmentation.fliplr(ybatch[0])


    visualizer.view_3_depths(ybatch[0], inv_depth, ybatchreverse[0])

    print('sum ', np.sum(ybatchreverse[0][np.isfinite(inv_depth)]-inv_depth[np.isfinite(inv_depth)]))

    model = Model(
        pretrained_net = 'vgg16',
        # some model specific dimension
        D = 1,
        input_dim = train_dataset.dim()
    )



    trainer = Trainer(
        model = model,
        args_adam = {"lr": 1e-4,
                     "betas": (0.9, 0.999),
                     "eps": 1e-8,
                     "weight_decay": 0.0},
        optim = torch.optim.Adam,
        loss_func = our_loss_fun,
        #augmentation_list = ['rotate180', 'fliplr', 'flipud', 'invert_colors'],
        augmentation_list=['fliplr', 'flipud'],
        reg = 1

    )



    xbatch1, ybatch1 = trainer.preprocess(xbatch, ybatch)
    #print("X batch shape, y batch shape after pre-processing : ", xbatch.shape, ybatch.shape)
    # fetch augmented tensors

    x = Variable(xbatch1, requires_grad = False)
    y = Variable(ybatch1, requires_grad = False)
    #print("X shape, y shape after conversion to variables : ", x, y)
    print("inverse_augmentation_func", trainer.inverse_augmentation_func)
    #inv_depth = trainer.inverse_augmentation_func[1](y[0])

    loss = our_loss_fun(y,y,trainer.reg, trainer.inverse_augmentation_func)

    #try with normal tensor vals
    #print('y[0][0][0]', y[0][0][0])
    #loss = torch.nn.MSELoss()(y[0][0],y[0][0])

    print("Loss: ", loss)
    print('mid')
    #visualizer.view_result(xbatch[0], inv_depth)
    print('end')