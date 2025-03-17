import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import BEEFISONet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy


EPSILON = 1e-8


class BEEFISO(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = BEEFISONet(args, False)
        self._snet = None
        self.logits_alignment = args["logits_alignment"]
        self.val_loader = None
        self.reduce_batch_size = args["reduce_batch_size"]
        self.random = args.get("random",None)
        self.imbalance = args.get("imbalance",None)

    def before_task(self):
        msg = self._network.convnets[0].load_state_dict(torch.load(self.args["pretrain_modelpath"]), strict=False)
        print(msg)
        for name, p in self._network.convnets[0].named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False

        for name, p in self._network.biases.named_parameters():
            p.requires_grad = True

        for name, p in self._network.forward_prototypes.named_parameters():
            p.requires_grad = True

        for name, p in self._network.new_fc.named_parameters():
            p.requires_grad = True

    def after_task(self):
        self._network_module_ptr.update_fc_after()
        self._known_classes = self._total_classes
        if self.reduce_batch_size:
            if self._cur_task == 0:
                self.args["batch_size"] = self.args["batch_size"]
            else:
                self.args["batch_size"] = self.args["batch_size"] * (self._cur_task+1) // (self._cur_task+2) 
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1 and self.args["is_compress"]:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc_before(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for id in range(self._cur_task):
                for p in self._network.convnets[id].parameters():
                    p.requires_grad = False
            for p in self._network.old_fc.parameters():
                p.requires_grad = False


        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )


        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        if self._cur_task > 0:
            if self.random or self.imbalance:
                val_dset = data_manager.get_finetune_dataset(known_classes=self._known_classes, total_classes=self._total_classes,
                                                         source="train", mode='train', appendent=self._get_memory(), type="ratio")
            else:
                _, val_dset = data_manager.get_dataset_with_split(np.arange(self._known_classes, self._total_classes),
                                                                       source='train', mode='train',
                                                                       appendent=self._get_memory(),
                                                                       val_samples_per_class=int(
                                                                           self.samples_old_class))
            self.val_loader = DataLoader(
                val_dset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader,self.val_loader)
        if self.random or self.imbalance:
            self.build_rehearsal_memory_imbalance(data_manager,self.samples_per_class)
        else:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, train_loader, test_loader, val_loader=None):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"]
            )
            self.epochs = self.args["init_epochs"]
            self.before_task()
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:

            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["expansion_epochs"]
            )
            
            self.epochs = self.args["expansion_epochs"]
            self.state = "expansion"
            for p in self._network.biases.parameters():
                p.requires_grad = False
            self._expansion(train_loader, test_loader, optimizer, scheduler)
            
            
            
            for p in self._network_module_ptr.forward_prototypes.parameters():
                p.requires_grad = False
            for p in self._network_module_ptr.backward_prototypes.parameters():
                p.requires_grad = False
            for p in self._network_module_ptr.new_fc.parameters():
                p.requires_grad = False
            for p in self._network_module_ptr.convnets[-1].parameters():
                p.requires_grad = False
            for p in self._network.biases.parameters():
                p.requires_grad = True
            self.state = "fusion"
            self.epochs = self.args["fusion_epochs"]
            self.per_cls_weights = torch.ones(self._total_classes).to(self._device)
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=0.05,
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            for n, p in self._network.named_parameters():
                if p.requires_grad == True:
                    print(n)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["fusion_epochs"]
            )
            self._fusion(val_loader,test_loader,optimizer,scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_en = 0.0
            correct, total = 0, 0
            for i, (_, inputs, inputs2, inputs_mviews, targets) in enumerate(train_loader):
                inputs, inputs2, inputs_mviews, targets = inputs.to(self._device, non_blocking=True), inputs2.to(self._device, non_blocking=True), inputs_mviews.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs_mviews = inputs_mviews.flatten(0, 1)
                output = self._network(inputs, inputs2, inputs_mviews)
                logits = F.softmax(output["logits"])

                loss_en = self.args["energy_weight"] * self.get_energy_loss(inputs, inputs2, inputs_mviews,targets,targets)
                loss = F.cross_entropy(logits, targets)
                loss = loss + loss_en
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_en += loss_en.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

    def _expansion(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_en = 0.0
            correct, total = 0, 0
            for i, (_, inputs, inputs2, inputs_mviews, targets) in enumerate(train_loader):
                inputs, inputs2, inputs_mviews, targets = inputs.to(self._device, non_blocking=True), inputs2.to(self._device, non_blocking=True), inputs_mviews.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

                if torch.isnan(inputs_mviews).any():
                    shiyan = 0

                inputs_mviews = inputs_mviews.flatten(0, 1)
                outputs = self._network(inputs, inputs2, inputs_mviews)
                logits,train_logits = (
                    F.softmax(outputs["logits"]),
                    F.softmax(outputs["train_logits"])
                )
                pseudo_targets = targets.clone()
                for task_id in range(self._cur_task+1):
                    if task_id == 0:
                        pseudo_targets = torch.where(targets<self.data_manager.get_accumulate_tasksize(task_id),task_id,pseudo_targets)
                    elif task_id == self._cur_task: 
                        pseudo_targets = torch.where(targets-self._known_classes+1>0,targets-self._known_classes+task_id,pseudo_targets)
                    else:
                        pseudo_targets = torch.where((targets<self.data_manager.get_accumulate_tasksize(task_id)) & (targets>self.data_manager.get_accumulate_tasksize(task_id-1)-1),task_id,pseudo_targets)

                train_logits_aligned = train_logits.clone()
                train_logits_aligned[:, list(range(self._cur_task))] /= self.logits_alignment
                loss_clf = F.cross_entropy(train_logits_aligned, pseudo_targets)
                loss_fe = torch.tensor(0.).cuda()
                loss_en = self.args["energy_weight"]  * self.get_energy_loss(inputs, inputs2, inputs_mviews,targets,pseudo_targets)
                loss = loss_clf + loss_fe + loss_en
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_en += loss_en.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_en {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_en / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
            
    def _fusion(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            # self.
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0
            for i, (_, inputs, inputs2, inputs_mviews, targets) in enumerate(train_loader):
                inputs, inputs2, inputs_mviews, targets = inputs.to(self._device, non_blocking=True), inputs2.to(self._device, non_blocking=True), inputs_mviews.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs_mviews = inputs_mviews.flatten(0, 1)
                outputs = self._network(inputs, inputs2, inputs_mviews)
                logits, train_logits = (
                    F.softmax(outputs["logits"]),
                    F.softmax(outputs["train_logits"])
                )
                
                loss_clf = F.cross_entropy(logits,targets)                
                loss_fe = torch.tensor(0.).cuda()
                loss_kd = torch.tensor(0.).cuda()     
                loss = loss_clf + loss_fe + loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                losses_kd += (
                    self._known_classes / self._total_classes
                ) * loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)


    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


    def get_energy_loss(self,inputs, inputs2, inputs_mviews,targets,pseudo_targets):
        inputs, inputs2, inputs_mviews = self.sample_q(inputs, inputs2, inputs_mviews)
        
        out = self._network(inputs, inputs2, inputs_mviews)
        if self._cur_task == 0:
            targets = targets + self._total_classes
            train_logits, energy_logits = out["logits"], out["energy_logits"]
        else:
            targets = targets + (self._total_classes - self._known_classes) + self._cur_task
            train_logits, energy_logits = out["train_logits"], out["energy_logits"]

        assert not torch.isnan(energy_logits).any()
        assert not torch.isnan(train_logits).any()


        logits = torch.cat([train_logits,energy_logits],dim=1)
        logits[:,pseudo_targets] = 1e-9
        energy_loss = F.cross_entropy(logits,targets)
        return energy_loss

    def sample_q(self, replay_buffer_inputs, replay_buffer_inputs2, replay_buffer_inputs_mviews, n_steps=3):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        self._network_copy = self._network_module_ptr.copy().freeze()

        init_sample_inputs = replay_buffer_inputs
        init_sample_inputs2 = replay_buffer_inputs2
        init_sample_inputs_mviews = replay_buffer_inputs_mviews
        # init_sample = torch.rot90(init_sample, 2, (2, 3))

        embedding_k_inputs = init_sample_inputs.clone().detach().requires_grad_(True)
        embedding_k_inputs2 = init_sample_inputs2.clone().detach().requires_grad_(True)

        assert not torch.isnan(init_sample_inputs_mviews).any()
        embedding_k_inputs_mviews = init_sample_inputs_mviews.clone().detach().requires_grad_(True)
        assert not torch.isnan(embedding_k_inputs_mviews).any()

        optimizer_gen_inputs = torch.optim.SGD([embedding_k_inputs], lr=1e-2)
        optimizer_gen_inputs2 = torch.optim.SGD([embedding_k_inputs2], lr=1e-2)
        optimizer_gen_inputs_mviews = torch.optim.SGD([embedding_k_inputs_mviews], lr=1e-2)

        for k in range(1, n_steps + 1):
            assert not torch.isnan(embedding_k_inputs_mviews).any()
            out = self._network_copy(embedding_k_inputs, embedding_k_inputs2, embedding_k_inputs_mviews)
            if self._cur_task == 0:
                energy_logits, train_logits = out["energy_logits"], out["logits"]
            else:
                energy_logits, train_logits = out["energy_logits"], out["train_logits"]
            num_forwards = energy_logits.shape[1]
            logits = torch.cat([train_logits,energy_logits],dim=1)
            negative_energy = torch.log(torch.sum(torch.softmax(logits,dim=1)[:,-num_forwards:]) + 1e-8)

            optimizer_gen_inputs.zero_grad()
            optimizer_gen_inputs2.zero_grad()
            optimizer_gen_inputs_mviews.zero_grad()

            negative_energy.sum().backward()

            optimizer_gen_inputs.step()
            embedding_k_inputs.data += 1e-3 * \
                torch.randn_like(embedding_k_inputs)
            optimizer_gen_inputs2.step()
            embedding_k_inputs2.data += 1e-3 * \
                                       torch.randn_like(embedding_k_inputs2)
            optimizer_gen_inputs_mviews.step()
            embedding_k_inputs_mviews.data += 1e-3 * \
                                        torch.randn_like(embedding_k_inputs_mviews)

        final_samples_inputs = embedding_k_inputs.detach()
        final_samples_inputs2 = embedding_k_inputs2.detach()
        final_samples_inputs_mviews = embedding_k_inputs_mviews.detach()

        return final_samples_inputs, final_samples_inputs2, final_samples_inputs_mviews
    
    
    def build_rehearsal_memory_imbalance(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified_imbalance(data_manager, per_class,self.random,self.imbalance)
        else:
            self._reduce_exemplar_imbalance(data_manager, per_class,self.random,self.imbalance)
            self._construct_exemplar_imbalance(data_manager, per_class,self.random,self.imbalance)
            
            
    def _reduce_exemplar_imbalance(self, data_manager, m,random,imbalance):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            l = sum(mask)
            if l == 0:
                continue
            if random or imbalance is not None:
                dd, dt = dummy_data[mask][:-1], dummy_targets[mask][:-1]
            else:
                dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_imbalance(self, data_manager, m, random=False,imbalance=None):
        increment  = self._total_classes - self._known_classes

        if  random:
            '''
            uniform random type
            '''
            selected_exemplars = []
            selected_targets = []
            logging.info("Contructing exmplars, totally random...({} total instances  {} classes)".format(increment*m, increment))
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(self._known_classes,self._total_classes),source="train",mode="test",ret_data=True)
            selected_indices = np.random.choice(list(range(len(data))),m*increment,repladce=False)
            for idx in selected_indices:
                selected_exemplars.append(data[idx])
                selected_targets.append(targets[idx])
            selected_exemplars = np.array(selected_exemplars)[:m*increment] 
            selected_targets = np.array(selected_targets)[:m*increment]
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if \
                    len(self._targets_memory) != 0 else selected_targets
        else:
            if imbalance is None:
                logging.info('Constructing exemplars...({} per classes)'.format(m))
                ms = np.ones(increment,dtype=int)*m
            elif imbalance>=1:
                '''
                half-half type
                '''
                ms=[m for _ in range(increment)]
                for i in range(increment//2):
                    ms[i]-=m//imbalance
                for i in range(increment//2,increment):
                    ms[i]+=m//imbalance
                np.random.shuffle(ms)
                ms = np.array(ms,dtype=int)
                logging.info("Constructing exmplars, Imbalance...({} or {} per classes)".format(m-m//imbalance,(m+m//imbalance)))
            elif imbalance<1: 
                '''
                exp type
                '''
                ms = np.array([imbalance**i for i in range(increment)])
                ms = ms/ms.sum()
                tot = m*increment
                ms = (tot*ms).astype(int)
                np.random.shuffle(ms)
                
            else:
                assert 0, "not implemented yet"
            logging.info("ms {}".format(ms))
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                      mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)

                # Select
                selected_exemplars = []
                exemplar_vectors = []  # [n, feature_dim]
                for k in range(1, ms[class_idx-self._known_classes]+1):
                    S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                    vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                    data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

                # uniques = np.unique(selected_exemplars, axis=0)
                selected_exemplars = np.array(selected_exemplars)
                if len(selected_exemplars)==0:
                    continue
                exemplar_targets = np.full(ms[class_idx-self._known_classes], class_idx)
                self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
                self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                    len(self._targets_memory) != 0 else exemplar_targets

                # Exemplar mean
                idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                       appendent=(selected_exemplars, exemplar_targets))
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4,pin_memory=True)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                self._class_means[class_idx, :] = mean
                # self._class_means[class_idx, :] = class_mean

    def _construct_exemplar_unified_imbalance(self, data_manager, m,random,imbalance):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))
        increment  = self._total_classes - self._known_classes

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            if sum(mask) == 0: continue
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        if  random:
            '''
            uniform sample type
            '''
            selected_exemplars = []
            selected_targets = []
            logging.info("Contructing exmplars, totally random...({} total instances  {} classes)".format(increment*m, increment))
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(self._known_classes,self._total_classes),source="train",mode="test",ret_data=True)
            selected_indices = np.random.choice(list(range(len(data))),m*increment,replace=False)
            for idx in selected_indices:
                selected_exemplars.append(data[idx])
                selected_targets.append(targets[idx])
            selected_exemplars = np.array(selected_exemplars) 
            selected_targets = np.array(selected_targets)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, selected_targets)) if \
                    len(self._targets_memory) != 0 else selected_targets
        else:
            if imbalance is None:
                logging.info('Constructing exemplars...({} per classes)'.format(m))
                ms = np.ones(increment,dtype=int)*m
            elif imbalance>=1:
                '''
                half-half type
                '''
                ms=[m for _ in range(increment)]
                for i in range(increment//2):
                    ms[i]-=m//imbalance
                for i in range(increment//2,increment):
                    ms[i]+=m//imbalance
                np.random.shuffle(ms)
                ms = np.array(ms,dtype=int)
                logging.info("Constructing exmplars, Imbalance...({} or {} per classes)".format(m-m//imbalance,(m+m//imbalance)))
            elif imbalance<1: 
                '''
                exp type
                '''
                ms = np.array([imbalance**i for i in range(increment)])
                ms = ms/ms.sum()
                tot = m*increment
                ms = (tot*ms).astype(int)
                np.random.shuffle(ms)
                
            else:
                assert 0, "not implemented yet"
            logging.info("ms {}".format(ms))
            # Construct exemplars for new classes and calculate the means
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                     mode='test', ret_data=True)
                class_loader = DataLoader(class_dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4,pin_memory=True)

                vectors, _ = self._extract_vectors(class_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)

                # Select
                selected_exemplars = []
                exemplar_vectors = []
                for k in range(1, ms[class_idx-self._known_classes]+1):
                    S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                    selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                    vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                    data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

                selected_exemplars = np.array(selected_exemplars)
                if len(selected_exemplars)==0:
                    continue
                exemplar_targets = np.full(ms[class_idx-self._known_classes], class_idx)
                self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                    else selected_exemplars
                self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                    len(self._targets_memory) != 0 else exemplar_targets

                # Exemplar mean
                exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                         appendent=(selected_exemplars, exemplar_targets))
                exemplar_loader = DataLoader(exemplar_dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(exemplar_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                _class_means[class_idx, :] = mean
                # _class_means[class_idx,:] = class_mean

            self._class_means = _class_means

