from helper.visualizer import save_result
import torch
import os
from helper.metrics import calculate_pixelwise_accuracy_saadhana
import numpy as np
from torch.autograd import Variable


class Tester(object):
    def __init__(self, model, test_loader, save_result_path, accuracy_thresholds, cuda):
        self.model = model
        self.test_loader = test_loader
        self.save_result_path = save_result_path
        self.accuracy_thresholds = accuracy_thresholds
        self.cuda = cuda

    def test(self):
        print("model testing started...")
        model = self.model
        test_scores = []
        for itr in range(300):
            image, actual_depth = self.test_loader.next_batch(batch_size = 1)
            image = torch.FloatTensor(np.array(image, dtype = float))
            actual_depth = torch.FloatTensor(np.array(actual_depth, dtype = float))
            if self.cuda:
                image = Variable(image.cuda(0))
                actual_depth = Variable(actual_depth.cuda(0))
            else:
                image = Variable(image)
                actual_depth = Variable(actual_depth)
            print("testing image " + str(itr))
            predicted_depth = model.forward(image)
            result_path = os.path.join(self.save_result_path, "result" + str(itr))
            test_scores.append(calculate_pixelwise_accuracy_saadhana(pred_depth = predicted_depth,
                                                                     actual_depth = actual_depth,
                                                                     accuracy_thresholds = self.accuracy_thresholds))
            save_result(image = image.cpu().data.numpy(), actual_depth = actual_depth.cpu().data.numpy(),
                        predicted_depth = predicted_depth.cpu().data.numpy(), file_name = result_path)
        test_acc = np.mean(test_scores)
        print("Test Accuracy : "+str(test_acc))
        print("model testing completed...")
