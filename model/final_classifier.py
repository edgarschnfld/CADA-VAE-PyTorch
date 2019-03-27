import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
from torch.nn import functional as F
import sys
from torch.utils.data import Dataset ,DataLoader
import numpy as np


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class CLASSIFIER:
    def __init__(self,model, _train_X, _train_Y,_test_seen_X,_test_seen_Y,_test_novel_X, _test_novel_Y, seenclasses,novelclasses,
                 _nclass, device , _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, use = None, ignore = None,train_only=False,test_only=False,do_nothing=False):

        self.train_only = train_only
        self.device = device
        print('DEVICE')
        print(self.device)

        self.train_X =  _train_X.to(self.device)
        self.train_Y = _train_Y.to(self.device)

        self.test_seen_feature = _test_seen_X.to(self.device)
        self.test_seen_label = _test_seen_Y.to(self.device)
        self.test_novel_feature = _test_novel_X.to(self.device)
        self.test_novel_label = _test_novel_Y.to(self.device)


        self.seenclasses = seenclasses.to(self.device)
        self.novelclasses = novelclasses.to(self.device)
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        print('self.input_dim')
        print(self.input_dim)

        self.average_loss = 0


        self.model = model.to(self.device)


        self.criterion = model.lossfunction      ######

        self.input = torch.FloatTensor(_batch_size, self.input_dim).to(self.device)
        self.label = torch.LongTensor(_batch_size).to(self.device)

        self.lr = _lr
        self.beta1 = _beta1

        f = list(filter(lambda x:  x.requires_grad, model.parameters()))
        self.optimizer = optim.Adam(f, lr=_lr, betas=(_beta1, 0.999))#

        self.criterion.to(self.device)
        self.input = self.input.to(self.device)
        self.label = self.label.to(self.device)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        self.loss = 0

        self.used_indices = torch.LongTensor([]).to(self.device)
        self.all_indices  = torch.linspace(0,self.ntrain-1,self.ntrain).long().to(self.device)

        self.current_epoch = 0

        self.acc_novel, self.acc_seen, self.H , self.acc = 0, 0, 0, 0

        self.intra_epoch_accuracies = [()]*10
        if do_nothing==False:
            if not generalized:
                print('...')

            if test_only==False:
                if generalized:
                    self.acc_seen, self.acc_novel, self.H = self.fit()
                else:
                    self.acc = self.fit_zsl()

            else:
                if generalized:
                    best_H = -1
                    with torch.no_grad():
                        acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
                        acc_novel = self.val_gzsl(self.test_novel_feature, self.test_novel_label, self.novelclasses)

                    if (acc_seen+acc_novel)>0:
                        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
                    else:
                        H = 0

                    if H > best_H:

                        best_seen = acc_seen
                        best_novel = acc_novel
                        best_H = H
                    self.acc_seen, self.acc_novel, self.H =best_seen,best_novel,best_H

                else:
                    with torch.no_grad():
                        acc = self.val(self.test_novel_feature, self.test_novel_label, self.novelclasses)

                    self.acc = acc

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):

                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)


                inputv = self.input
                labelv = self.label

                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()#data[0]
                loss.backward()
                self.optimizer.step()

            self.current_epoch +=1

            acc = 0
            if self.train_only==False:
                with torch.no_grad():
                    acc = self.val(self.test_novel_feature, self.test_novel_label, self.novelclasses)

            if acc > best_acc:
                best_acc = acc

            self.loss = loss

        return best_acc

    def fit(self):
        best_H = -1
        best_seen = 0
        best_novel = 0


        Dataset = TrainDataset(self.train_X,self.train_Y)
        dataloader = DataLoader(Dataset, batch_size=self.batch_size,
                        shuffle=True,drop_last=True)#, num_workers=1)

        iterations_per_epoch = int(self.ntrain/self.batch_size)

        checkpoints = torch.linspace(0,iterations_per_epoch,12)
        checkpoints = checkpoints[1:-1]
        checkpoints = [int(x) for x in checkpoints]


        for epoch in range(self.nepoch):

            self.average_loss = 0

            i = 0
            c = 0
            for batch in dataloader:
                self.model.zero_grad()

                output = self.model(batch['x'])

                loss = self.criterion(output, batch['y'])

                loss.backward()
                #print(loss)

                if i>0.8*iterations_per_epoch:

                    self.average_loss += loss.item()/(0.2*iterations_per_epoch)


                self.optimizer.step()

                i+=1



            acc_seen = 0
            acc_novel = 0

            self.current_epoch +=1

            if self.train_only==False:
                with torch.no_grad():
                    acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
                    acc_novel = self.val_gzsl(self.test_novel_feature, self.test_novel_label, self.novelclasses)

            if (acc_seen+acc_novel)>0:
                H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
            else:
                H = 0

            if H > best_H:

                best_seen = acc_seen
                best_novel = acc_novel
                best_H = H


        self.loss = loss
        return best_seen, best_novel, best_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)#.to(self.device)###added cuda()
            self.train_X = self.train_X[perm ]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)#.to(self.device)###added cuda()
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes):

        with torch.no_grad():
            start = 0
            ntest = test_X.size()[0]
            predicted_label = torch.LongTensor(test_label.size())
            for i in range(0, ntest, self.batch_size):

                end = min(ntest, start+self.batch_size)

                output = self.model(test_X[start:end]) #.to(self.device)

                #_, predicted_label[start:end] = torch.max(output.data, 1)
                predicted_label[start:end] = torch.argmax(output.data, 1)

                start = end

            #print(str(predicted_label[:3]).ljust(40,'.'), end= ' '     )
            acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
            return acc


    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):

        per_class_accuracies = Variable(torch.zeros(target_classes.size()[0]).float().to(self.device)).detach()

        predicted_label = predicted_label.to(self.device)

        for i in range(target_classes.size()[0]):

            is_class = test_label==target_classes[i]

            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())

        return per_class_accuracies.mean()


    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)

            output = self.model(test_X[start:end].to(self.device))

            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))

        return acc


    def compute_per_class_acc(self, test_label, predicted_label, nclass):

        per_class_accuracies = torch.zeros(nclass).float().to(self.device).detach()

        target_classes = torch.arange(0, nclass, out=torch.LongTensor()).to(self.device) #changed from 200 to nclass on 24.06.
        predicted_label = predicted_label.to(self.device)
        test_label = test_label.to(self.device)

        for i in range(nclass):

            is_class = test_label==target_classes[i]

            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())

        return per_class_accuracies.mean()


class TrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_X, train_Y):

        self.train_X = train_X
        self.train_Y = train_Y.long()

    def __len__(self):
        return self.train_X.size(0)

    def __getitem__(self, idx):

        return {'x': self.train_X[idx,:], 'y': self.train_Y[idx] }
