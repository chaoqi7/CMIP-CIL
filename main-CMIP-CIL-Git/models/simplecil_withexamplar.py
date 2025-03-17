'''
Re-implementation of SimpleCIL (https://arxiv.org/abs/2303.07338) without pre-trained weights. 
The training process is as follows: train the model with cross-entropy in the first stage and replace the classifier with prototypes for all the classes in the subsequent stages. 
Please refer to the original implementation (https://github.com/zhoudw-zdw/RevisitingCIL) if you are using pre-trained weights.
'''
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleCosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size = 16
# milestones = [80, 120]
milestones = [16, 24]

class SimPlecil_withExamplar(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleCosineIncrementalNet(args, False)
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def before_task(self):
        msg = self._network.convnet.load_state_dict(torch.load(self.args["pretrain_modelpath"]), strict=False)
        print(msg)
        for name, p in self._network.convnet.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False

        for name, p in self._network.fc.named_parameters():
            p.requires_grad = True

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self, trainloader: object, model: object, args: object) -> object:
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, data2, data_mviews, label) = batch
                data = data.cuda()
                data2 = data2.cuda()
                data_mviews = data_mviews.cuda()
                label = label.cuda()
                data_mviews = data_mviews.flatten(0, 1)
                embedding = model(data, data2, data_mviews)["img_feats"]

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", appendent=self._get_memory(),)
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", appendent=self._get_memory(),)
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=self.args["init_lr"],
            weight_decay=self.args["init_weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args['init_epoch'], eta_min=self.min_lr
        )

        if self._cur_task == 0:
            self.before_task()
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, inputs2, inputs_mviews, targets) in enumerate(train_loader):
                inputs, inputs2, inputs_mviews, targets = inputs.to(self._device), inputs2.to(self._device), inputs_mviews.to(self._device), targets.to(self._device)
                inputs_mviews = inputs_mviews.flatten(0, 1)
                output = self._network(inputs, inputs2, inputs_mviews, layerout = False, test=False)
                logits = F.softmax(output["logits"])

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)
    

   
