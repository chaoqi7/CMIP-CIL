import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 30
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 170
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 16
weight_decay = 2e-4
num_workers = 8
T = 2

class CrossTest(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

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


    def incremental_train(self, data_manager):
        _cur_task = 0
        self._total_classes = self._known_classes + data_manager.get_task_size(_cur_task)
        #self._network.last_cls = data_manager.get_task_size(-1) # modified by Qi
        #self._network.nb_tasks = data_manager.nb_tasks # modified by Qi
        self._network.update_fc(self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=init_lr,
            weight_decay=init_weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
        )
        self.before_task()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, inputs2, inputs_mviews, targets) in enumerate(train_loader):
                inputs, inputs2, inputs_mviews, targets = inputs.to(self._device), inputs2.to(self._device), inputs_mviews.to(self._device), targets.to(self._device)
                inputs_mviews = inputs_mviews.flatten(0, 1)
                output = self._network(inputs, inputs2, inputs_mviews, test=False)
                logits = F.softmax(output["logits"])

                # point_feats = output["point_feats"]
                # point_feats2 = output["point_feats2"]
                # img_feats = output["img_feats"]

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
                    'CrossTest',
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
                print(info)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    'CrossTest',
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
                prog_bar.set_description(info)
        logging.info(info)