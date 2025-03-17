import logging
import numpy as np
import utils.data_utils as d_utils
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, ModelNet, ShapeNet, ScanObject, M_modelnet40, M_shapenet55, M_scanobject
from tqdm import tqdm
import os
import torch
import pickle

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args, _trsf = True, join_thelast = False):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed, trsf= _trsf)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            if 'shapenet55' in dataset_name:
                if offset < increment:
                    self._increments[-1] = self._increments[-1] + offset
                else:
                    self._increments.append(offset)
            else:
                self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None, old_data = None, cur_task = 0
    ):
        if source == "train":
            x, x_mviews, y = self._train_data, self._train_data_mviews, self._train_targets
        elif source == "test":
            x, x_mviews, y = self._test_data, self._test_data_mviews, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        try:
            if mode == "train" or mode == "test":
                # trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
                trans_pcs_1 = transforms.Compose(
                    [
                        d_utils.PointcloudToTensor(),
                        d_utils.PointcloudNormalize(),
                        d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                        d_utils.PointcloudRotate(),
                        d_utils.PointcloudTranslate(0.5, p=1),
                        d_utils.PointcloudJitter(p=1),
                        d_utils.PointcloudRandomInputDropout(p=1),
                    ])

                trans_pcs_2 = transforms.Compose(
                    [
                        d_utils.PointcloudToTensor(),
                        d_utils.PointcloudNormalize(),
                        d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                        d_utils.PointcloudRotate(),
                        d_utils.PointcloudTranslate(0.5, p=1),
                        d_utils.PointcloudJitter(p=1),
                        d_utils.PointcloudRandomInputDropout(p=1),
                    ])

                trans_mviews = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            elif mode == "flip":
                trsf = transforms.Compose(
                    [
                        *self._test_trsf,
                        transforms.RandomHorizontalFlip(p=1.0),
                        *self._common_trsf,
                    ]
                )

            # elif mode == "test":
            #    trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
            else:
                raise ValueError("Unknown mode {}.".format(mode))
        except Exception:
            trsf = None

        data, data_mviews, targets = [], [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_data_mviews, class_targets = self._select(
                    x, x_mviews, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_data_mviews, class_targets = self._select_rmm(
                    x, x_mviews, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            if len(class_data) != len(class_data_mviews):
                shiyan = 0
            data.append(class_data)
            data_mviews.append(class_data_mviews)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_data_mviews, appendent_targets = appendent
            data.append(appendent_data)
            data_mviews.append(appendent_data_mviews)
            targets.append(appendent_targets)

        data, data_mviews, targets = np.concatenate(data), [item for sublist in data_mviews for item in sublist], np.concatenate(targets)

        if self.args['pretrain'] == True:
            self.use_path = True
        if ret_data:
            return data,  data_mviews, targets, DummyDataset(data, data_mviews, targets, trans_pcs_1, trans_pcs_2, trans_mviews, self.use_path)
        else:
            return DummyDataset(data, data_mviews, targets, trans_pcs_1, trans_pcs_2, trans_mviews, self.use_path)

        
    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, x_mviews, y = self._train_data, self._train_data_mviews, self._train_targets
        elif source == "test":
            x, x_mviews, y = self._test_data, self._test_data_mviews, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        try:
            if mode == "train" or mode == "test":
                trans_pcs_1 = transforms.Compose(
                    [
                        d_utils.PointcloudToTensor(),
                        d_utils.PointcloudNormalize(),
                        d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                        d_utils.PointcloudRotate(),
                        d_utils.PointcloudTranslate(0.5, p=1),
                        d_utils.PointcloudJitter(p=1),
                        d_utils.PointcloudRandomInputDropout(p=1),
                    ])

                trans_pcs_2 = transforms.Compose(
                    [
                        d_utils.PointcloudToTensor(),
                        d_utils.PointcloudNormalize(),
                        d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                        d_utils.PointcloudRotate(),
                        d_utils.PointcloudTranslate(0.5, p=1),
                        d_utils.PointcloudJitter(p=1),
                        d_utils.PointcloudRandomInputDropout(p=1),
                    ])

                trans_mviews = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            else:
                raise ValueError("Unknown mode {}.".format(mode))
        except Exception:
            trsf = None

        train_data, train_data_mviews, train_targets = [], [], []
        val_data, val_data_mviews, val_targets = [], [], []
        for idx in indices:
            class_data, class_data_mviews, class_targets = self._select(
                x, x_mviews, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_data_mviews.append([class_data_mviews[i] for i in val_indx])
            #.append(class_data_mviews[val_indx])
            val_targets.append(class_targets[val_indx])

            train_data.append(class_data[train_indx])
            train_data_mviews.append([class_data_mviews[i] for i in train_indx])
            # train_data_mviews.append(class_data_mviews[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_data_mviews, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_data_mviews, append_targets = self._select(
                    appendent_data, appendent_data_mviews, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                # val_data_mviews.append(append_data_mviews[val_indx])
                val_data_mviews.append([append_data_mviews[i] for i in val_indx])
                val_targets.append(append_targets[val_indx])

                train_data.append(append_data[train_indx])
                # train_data_mviews.append(append_data_mviews[train_indx])
                train_data_mviews.append([append_data_mviews[i] for i in train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_data_mviews, train_targets = np.concatenate(train_data), [item for sublist in train_data_mviews for item in sublist], np.concatenate(
            train_targets
        )
        val_data, val_data_mviews, val_targets = np.concatenate(val_data), [item for sublist in val_data_mviews for item in sublist], np.concatenate(val_targets)

        return DummyDataset(train_data, train_data_mviews, train_targets, trans_pcs_1, trans_pcs_2, trans_mviews, self.use_path), DummyDataset(val_data, val_data_mviews, val_targets, trans_pcs_1, trans_pcs_2, trans_mviews, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed, trsf = True):
        idata = _get_idata(dataset_name, self.args)

        if self.args['pretrain'] == False:
            idata.download_data()
        else:
            idata.download_pretraindata()

        # Data
        self._train_data, self._train_data_mviews, self._train_targets = idata.train_data, idata.train_data_mviews, idata.train_targets
        self._test_data, self._test_data_mviews, self._test_targets = idata.test_data, idata.test_data_mviews, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        if trsf:
            self._train_trsf = idata.train_trsf
            self._test_trsf = idata.test_trsf
            self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, x_mviews, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], [x_mviews[i] for i in idxes], y[idxes]
        '''
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]
        '''

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, pcs, data_mviews, labels, trans_pcs_1, trans_pcs_2, trans_mviews, use_path=False):
        assert len(pcs) == len(pcs), "Data size error!"
        self.pcs = pcs
        self.data_mviews = data_mviews
        self.labels = labels
        self.trans_pcs_1 = trans_pcs_1
        self.trans_pcs_2 = trans_pcs_2
        self.trans_mviews = trans_mviews
        self.use_path = use_path

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, idx):
        pcs1, pcs2, data_mviews, label = None, None, 10*[None], None
        if self.use_path:
            # if 'ModelNet40_Multimodals_RandMask' in self.pcs[0] or 'ModelNet40_Multimodals_RandMask' in self.pcs[0] : ## pre train for modelnet40
            if 'RandMask' in self.pcs[0]:
                with open(self.pcs[idx], 'rb') as f:
                    pcs, _ = pickle.load(f)

                pcs_numpy = pcs.numpy()
                if pcs_numpy.shape[0] > 1024:
                    pcs_indices = np.random.choice(pcs_numpy.shape[0], 1024, replace=False)
                    pcs_numpy = pcs_numpy[pcs_indices]
                if pcs_numpy.shape[0] < 1024:
                    pcs_indices = np.random.choice(pcs_numpy.shape[0], 1024, replace=True)
                    pcs_numpy = pcs_numpy[pcs_indices]

                pcs1 = self.trans_pcs_1(pcs_numpy)
                pcs2 = self.trans_pcs_2(pcs_numpy)
                png_names = self.data_mviews[idx]
                data_mviews = [self.trans_mviews(Image.open(png_names[i])) for i in range(10)]
                data_mviews = torch.stack(data_mviews)
                label = self.labels[idx]

                return idx, pcs1, pcs2, data_mviews, label
            else:
                pcs = np.load(self.pcs[idx])  # self.trsf(pil_loader(self.images[idx]))
                pcs1 = self.trans_pcs_1(pcs)
                pcs2 = self.trans_pcs_2(pcs)
                data_multiviews = []
                png_names = os.listdir(self.data_mviews[idx])
                for i in range(10):
                    png_name = png_names[i].decode('utf-8')
                    if png_name.endswith(".png"):
                        # data_multiviews.append(np.expand_dims(np.array(Image.open(os.path.join(self.data_mviews[idx], item))), axis = 0))# pil_loader(self.images[idx])
                        data_mviews[i] = self.trans_mviews(Image.open(os.path.join(self.data_mviews[idx], png_name)))
                data_mviews = torch.stack(data_mviews)
                label = self.labels[idx]
                return idx, pcs1, pcs2, data_mviews, label
        else:
            try:
                pcs1 = self.trans_pcs_1(self.pcs[idx])
                pcs2 = self.trans_pcs_2(self.pcs[idx])
                for i in range(10):
                    if isinstance(self.data_mviews[idx][i], np.ndarray):
                        img = Image.fromarray(self.data_mviews[idx][i].astype(np.uint8), mode='RGB')
                    else:
                        img = self.data_mviews[idx][i]
                    data_mviews[i] = self.trans_mviews(img)
                data_mviews = torch.stack(data_mviews)
                label = self.labels[idx]
            except  Exception as e:
                print(e)
            # data_multiviews = (data_multiviews / 255.0).astype(np.float32)
            # data_multiviews = data_multiviews.transpose((0, 3, 1, 2))
            return idx, pcs1, pcs2, data_mviews, label

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "modelnet40":
        return ModelNet()
    elif name == "shapenet55":
        return ShapeNet()
    elif name == "scanobject":
        return ScanObject()
    elif name == "m_modelnet40":
        return M_modelnet40()
    elif name == "m_shapenet55":
        return M_shapenet55()
    elif name == "m_scanobject":
        return M_scanobject()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
