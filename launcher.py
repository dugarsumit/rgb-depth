from architecture.model import Model
from architecture.trainer import Trainer
from helper.metrics import our_loss_fun
from architecture.tester import Tester
from helper.rmrc import Rmrc
import os
import dateutil.tz
import datetime
import torch


if __name__ == "__main__":

    import torch.cuda as cutorch
    import time

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M')
    # comment this path and create a new vaiable according to your own path.

    base_path = "data/"
    #base_path = "/home/saadhana/Semester2/DLCV_Data/low_resolution"

    rmrc_train_data_path = os.path.join(base_path, "train")
    rmrc_val_data_path = os.path.join(base_path, "val")
    rmrc_test_data_path = os.path.join(base_path, "test")

    train_dataset = Rmrc(data_path = rmrc_train_data_path)
    val_dataset = Rmrc(data_path = rmrc_val_data_path)
    test_dataset = Rmrc(data_path = rmrc_test_data_path)

    exp_name = "rmrc_%s" % timestamp
    loss_graph_name = "loss_%s" % timestamp
    acc_graph_name = "acc_%s" % timestamp

    checkpoint_path = "checkpoint/"
    documents_path = "documents/"
    result_path = "results/"
    save_model_path = os.path.join(checkpoint_path, exp_name)

    save_result_path = os.path.join(result_path, exp_name)
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    result_file = open(os.path.join(save_result_path, "workfile"), "w")
    loss_graph_path = os.path.join(save_result_path, "loss_graph")
    acc_graph_path = os.path.join(save_result_path, "acc_graph")
    result_img_path = save_result_path

    # path of the model that need to be used

    # load_model_path = os.path.join(checkpoint_path, "rmrc_2017_07_29_23_40") #Neha - sumits
    # load_model_path = os.path.join(checkpoint_path, "rmrc_2017_07_30_20_09") #Neha - sumits + 200 + 200
    # load_model_path = os.path.join(checkpoint_path, "rmrc_2017_07_30_21_17")  # Neha - sumits + 200 + 200 + 200 (reg=0.5)
    # load_model_path = os.path.join(checkpoint_path, "rmrc_2017_07_30_21_22")  # Neha - sumits + 200 + 200 + 200 (reg=0.5) + 200 (reg=0.5)

    # load_model_path = os.path.join(checkpoint_path, "rmrc_2017_07_30_22_16_Sumit") # Sumit - 29_23_40 + 4000 * 12
    load_model_path = os.path.join(checkpoint_path, "rmrc_2017_08_04_00_12_epoch4_rootmeansquare")
    test_model_path = load_model_path

    use_saved_model = False
    cuda_flag = False
    train_upsampler = False
    train_upsampler_present = False
    save_model = False
    use_set_loss = True
    #result_file = None
    overfit = True
    randomize_batch = False
    hyperparameter_evaluation = False
    start_testing = False

    #print('CUDA MEM: ', cutorch.getMemoryUsage(0))
    startModelInit = time.time()
    if use_saved_model:
        if cuda_flag:
            model = torch.load(load_model_path)
            model = model.cuda(0)
        else:
            model = torch.load(load_model_path, map_location = lambda storage, loc: storage)

        model.flag = train_upsampler
        model.save_model = save_model
        if(train_upsampler_present == False and train_upsampler == True):
            # We want to add the upsampler scale(scale 3) to our model only when the scale 3
            # is not present. We may encounter situations in which we want to train our model
            # from a checkpoint where scale 3 is already present
            model.add_upsampler_2_model(cuda=cuda_flag)

    else:
        model = Model(
            pretrained_net = 'vgg16',
            # some model specific dimension
            D = 1,
            input_dim = train_dataset.dim(),
            train_upsampler = train_upsampler,
            save_model = save_model,
            cuda=cuda_flag
        )
    startTrainerInit = time.time()
    print('Model initialization completed in: ',startTrainerInit-startModelInit)

    if(hyperparameter_evaluation):
        from architecture.trainer_hyperparameter_version import Trainer as TrHyper

        weight_decays = [10,5,1,0.5,0.1,0.05,0]

        for it in range(len(weight_decays)):
            save_result_path = os.path.join(result_path, exp_name, '_weight_'+str(weight_decays[it]))
            if not os.path.exists(save_result_path):
                os.makedirs(save_result_path)
            result_file = open(os.path.join(save_result_path, "workfile"), "w")
            loss_graph_path = os.path.join(save_result_path, "loss_graph")
            acc_graph_path = os.path.join(save_result_path, "acc_graph")
            result_img_path = save_result_path
            startTrainerInit = time.time()
            trainer = TrHyper(
                model=model,
                args_adam={"lr": 1e-4,
                           "betas": (0.9, 0.999),
                           "eps": 1e-8,
                           "weight_decay": 10},
                optim=torch.optim.Adam,
                loss_func=our_loss_fun,
                augmentation_list=['flipud', 'fliplr'],
                reg=1.5,
                train_upsampler=train_upsampler,
                cuda=cuda_flag,
                accuracy_thresholds=[1.25, 1.25 * 1.25, 1.25 * 1.25 * 1.25],
                layer_hyper_params={'weight_decay': weight_decays[it]}
            )
            startTraining = time.time()
            print('Trainer initialization completed in: ', startTraining - startTrainerInit)
            trainer.train(
                num_epochs=5,
                nth=1,
                train_loader=train_dataset,
                val_loader=val_dataset,
                save_model_path=save_model_path+'_weight_'+str(weight_decays[it]),
                loss_graph_path=loss_graph_path,
                acc_graph_path=acc_graph_path,
                result_img_path=result_img_path,
                cuda=cuda_flag,
                batch_size=1,
                num_iterations=4000,
                use_set_loss=use_set_loss,
                result_file=result_file,
                overfit=overfit,
                randomize_batch=randomize_batch
            )
            endTraining = time.time()
            print('Training completed in: ', endTraining - startTraining)

    else:
        trainer = Trainer(
            model = model,
            args_adam = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 10},
            optim = torch.optim.Adam,
            loss_func = our_loss_fun,
            augmentation_list = ['flipud', 'fliplr'],
            reg = 1.5,
            train_upsampler = train_upsampler,
            cuda = cuda_flag,
            accuracy_thresholds = [1.25, 1.25*1.25, 1.25*1.25*1.25],
        )

        tester = Tester(
            model = model,
            test_loader = test_dataset,
            save_result_path = save_result_path,
            accuracy_thresholds = [1.25, 1.25 * 1.25, 1.25 * 1.25 * 1.25],
            cuda = cuda_flag
        )

        startTraining = time.time()
        print('Trainer initialization completed in: ', startTraining-startTrainerInit)
        if start_testing:
            tester.test()
        else:
            trainer.train(
                num_epochs = 1,
                nth = 1,
                train_loader = train_dataset,
                val_loader = val_dataset,
                save_model_path = save_model_path,
                loss_graph_path = loss_graph_path,
                acc_graph_path = acc_graph_path,
                result_img_path = result_img_path,
                cuda = cuda_flag,
                batch_size = 1,
                num_iterations = 1,
                use_set_loss = use_set_loss,
                result_file = result_file,
                overfit = overfit,
                randomize_batch = randomize_batch
            )
        endTraining = time.time()
        print('Training completed in: ', endTraining-startTraining)





