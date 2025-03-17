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
from lightly.loss.ntx_ent_loss import NTXentLoss
import os

EPSILON = 1e-8

pertrain_epoch = 100
pertrain_lr = 0.001
init_milestones = [30, 60, 85]
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
save_path = os.path.join(f'/root/autodl-tmp/PyCIL_CrossModal', 'pretrain')
# save_file = os.path.join(f'/root/autodl-tmp/Ease_CrossModal_Code/experiments/Crossmodel_Pretrain/', 'best_model.pth')

class PreTrain(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.criterion = NTXentLoss(temperature=0.02).cuda() # 0.04, 0.01, 0.005

    def pretrain(self, data_manager):
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        #self._network.last_cls = data_manager.get_task_size(-1) # modified by Qi
        #self._network.nb_tasks = data_manager.nb_tasks # modified by Qi
        # self._network.update_fc(self._total_classes)
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader):
        self._network.to(self._device)
        #if self._old_network is not None:
        #    self._old_network.to(self._device)
        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=pertrain_lr,
            weight_decay=init_weight_decay,
        )
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=pertrain_epoch, eta_min=0)

        self._init_train(train_loader, optimizer, scheduler)
        '''
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
        '''

    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(pertrain_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses_imid = 0.0
            losses_cmid = 0.0
            losses = 0.0

            for i, (_, inputs, inputs2, inputs_mviews, targets) in enumerate(train_loader):
                inputs, inputs2, inputs_mviews, targets = inputs.to(self._device), inputs2.to(self._device), inputs_mviews.to(self._device), targets.to(self._device)
                inputs_mviews = inputs_mviews.flatten(0, 1)
                point_feats, point_feats2, img_feats = self._network.convnet(inputs, inputs2, inputs_mviews)

                loss_imid = self.criterion(point_feats, point_feats2)
                point_feats = torch.stack([point_feats, point_feats2]).mean(dim=0)
                loss_cmid = self.criterion(point_feats, img_feats)
                loss = loss_imid + loss_cmid
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_imid += loss_imid.item()
                losses_cmid += loss_cmid.item()
                losses += loss.item()

            scheduler.step()
            info = "Epo{}/{} => Loss {:.2f}, Loss_imid {:.2f}, Loss_cmid {:.2f}".format(
                epoch + 1,
                pertrain_epoch,
                losses / len(train_loader),
                losses_imid / len(train_loader),
                losses_cmid / len(train_loader),
            )

            prog_bar.set_description(info)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_model_path = os.path.join(save_path, self.args['dataset'] + '_' +str(self.args['seed'][0]))
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)

            torch.save(self._network.convnet.state_dict(), os.path.join(save_model_path, 'best_model_epo'+ str(epoch )+'.pth'))
            logging.info(info)
