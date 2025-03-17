import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
from tqdm import tqdm
import pickle
import h5py
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNet(iData):
    use_path = False

    def download_data(self, config = None):
        self.root = '/root/autodl-tmp/DataSet/modelnet40_normal_resampled/' #config.DATA_PATH
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.uniform = True
        self.process_data = False
        self.npoints = 1024# config.N_POINTS
        self.use_normals = False #config.USE_NORMALS
        self.num_category = 40 #config.NUM_CATEGORY

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        shape_names_train = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['train']]
        self.datapath_train = [(shape_names_train[i], os.path.join(self.root, shape_names_train[i], shape_ids['train'][i]) + '.txt') for i
                         in range(len(shape_ids['train']))]

        shape_names_test = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['test']]
        self.datapath_test = [(shape_names_test[i], os.path.join(self.root, shape_names_test[i], shape_ids['test'][i]) + '.txt') for i
                         in range(len(shape_ids['test']))]

        print('The size of %s data is %d' % ('train', len(self.datapath_train)))
        print('The size of %s data is %d' % ('test', len(self.datapath_test)))

        if self.uniform:
            self.train_save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        else:
            self.train_save_path = os.path.join(self.root,'modelnet%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))

        if (not os.path.exists(self.train_save_path)) or (not os.path.exists(self.test_save_path)):
            self.process_data = True

        if self.process_data:
            print('Processing data (only running in the first time)')
            # print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
            self.train_data = [None] * len(self.datapath_train)
            self.train_targets = [None] * len(self.datapath_train)
            self.test_data = [None] * len(self.datapath_test)
            self.test_targets = [None] * len(self.datapath_test)

            for index in tqdm(range(len(self.datapath_train)), total=len(self.datapath_train)):
                fn = self.datapath_train[index]
                cls = self.classes[self.datapath_train[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.train_data[index] = point_set
                self.train_targets[index] = cls

            with open(self.train_save_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_targets], f)

            for index in tqdm(range(len(self.datapath_test)), total=len(self.datapath_test)):
                fn = self.datapath_test[index]
                cls = self.classes[self.datapath_test[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.test_data[index] = point_set
                self.test_targets[index] = cls

            with open(self.test_save_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_targets], f)
        else:
            print('Load processed data')
            with open(self.train_save_path, 'rb') as f:
                self.train_data, self.train_targets = pickle.load(f)
            with open(self.test_save_path, 'rb') as f:
                self.test_data, self.test_targets = pickle.load(f)
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)
        if not self.use_normals:
            self.train_data = self.train_data[:,:, 0:3]
            self.test_data = self.test_data[:, :, 0:3]
        self.train_data = self.train_data.transpose(0, 2, 1)
        self.test_data = self.test_data.transpose(0, 2, 1)
        print('Data Loading Finished..')

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

class ShapeNet(iData):
    use_path = False

    def download_data(self, config = None):
        self.root = '/root/autodl-tmp/DataSet/ShapeNet55/' #config.DATA_PATH
        self.uniform = True
        self.process_data = False
        self.npoints = 1024  # config.N_POINTS
        self.use_normals = False  # config.USE_NORMALS
        self.num_category = 55  # config.NUM_CATEGORY

        cLass_names = set()
        files = os.listdir(os.path.join(self.root, 'shapenet_pc'))
        for file in files:
            prefix = file.split('-')[0]
            cLass_names.add(prefix)
        self.cat = list(cLass_names)
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]

        shape_names_train = [x for x in shape_ids['train']]
        self.datapath_train = [(shape_names_train[i], os.path.join(self.root, 'shapenet_pc' , shape_names_train[i])) for i
            in range(len(shape_ids['train']))]

        shape_names_test = [x for x in shape_ids['test']]
        self.datapath_test = [(shape_names_test[i], os.path.join(self.root, 'shapenet_pc' , shape_names_test[i])) for i
            in range(len(shape_ids['test']))]

        print('The size of %s data is %d' % ('train', len(shape_ids['train'])))
        print('The size of %s data is %d' % ('test', len(shape_ids['test'])))

        if self.uniform:
            self.train_save_path = os.path.join(self.root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        else:
            self.train_save_path = os.path.join(self.root,'shapenet%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'shapene%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))

        if (not os.path.exists(self.train_save_path)) or (not os.path.exists(self.test_save_path)):
            self.process_data = True

        if self.process_data:
            print('Processing data (only running in the first time)')
            # print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
            self.train_data = [None] * len(self.datapath_train)
            self.train_targets = [None] * len(self.datapath_train)
            self.test_data = [None] * len(self.datapath_test)
            self.test_targets = [None] * len(self.datapath_test)

            for index in tqdm(range(len(self.datapath_train)), total=len(self.datapath_train)):
                fn = self.datapath_train[index]
                cls = self.classes[self.datapath_train[index][0].split('-')[0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.load(fn[1]).astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.train_data[index] = point_set
                self.train_targets[index] = cls

            with open(self.train_save_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_targets], f)

            for index in tqdm(range(len(self.datapath_test)), total=len(self.datapath_test)):
                fn = self.datapath_test[index]
                cls = self.classes[self.datapath_test[index][0].split('-')[0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.load(fn[1]).astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                if not self.use_normals:
                    point_set = point_set[:, 0:3]
                self.test_data[index] = point_set
                self.test_targets[index] = cls

            with open(self.test_save_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_targets], f)
        else:
            print('Load processed data')
            with open(self.train_save_path, 'rb') as f:
                self.train_data, self.train_targets = pickle.load(f)
            with open(self.test_save_path, 'rb') as f:
                self.test_data, self.test_targets = pickle.load(f)
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)
        if not self.use_normals:
            self.train_data = self.train_data[:,:, 0:3]
            self.test_data = self.test_data[:, :, 0:3]
        self.train_data = self.train_data.transpose(0, 2, 1)
        self.test_data = self.test_data.transpose(0, 2, 1)
        print('Data Loading Finished..')

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

class ScanObject(iData):
    use_path = False

    def download_data(self, config = None):
        self.root = '/root/autodl-tmp/DataSet/ScanObjectNN_h5_files/h5_files/main_split_nobg/'
        self.npoints = 1024  # config.N_POINTS
        self.uniform = True
        self.process_data = False
        self.num_category = 15

        self.train_load_path = os.path.join(self.root, 'training_objectdataset_augmentedrot.h5')
        self.test_load_path = os.path.join(self.root, 'test_objectdataset_augmentedrot.h5')

        if self.uniform:
            self.train_save_path = os.path.join(self.root, 'scanobject%d_%s_%dpts_fps.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'scanobject%d_%s_%dpts_fps.dat' % (self.num_category, 'test', self.npoints))
        else:
            self.train_save_path = os.path.join(self.root,'scanobject%d_%s_%dpts.dat' % (self.num_category, 'train', self.npoints))
            self.test_save_path = os.path.join(self.root, 'scanobject%d_%s_%dpts.dat' % (self.num_category, 'test', self.npoints))

        if (not os.path.exists(self.train_save_path)) or (not os.path.exists(self.test_save_path)):
            self.process_data = True

        if self.process_data:
            print('Processing data (only running in the first time)')
            train_file = h5py.File(self.train_load_path, 'r')
            test_file = h5py.File(self.test_load_path, 'r')
            self.train_data = [None] * len(train_file['data'])
            self.train_targets = [None] * len(train_file['label'])
            self.test_data = [None] * len(test_file['data'])
            self.test_targets = [None] * len(test_file['label'])
            train_data_tmpt = np.array(train_file['data']).astype(np.float32)
            train_targets_tmpt = np.array(train_file['label']).astype(int)
            test_data_tmpt = np.array(test_file['data']).astype(np.float32)
            test_targets_tmpt = np.array(test_file['label']).astype(int)

            for index in tqdm(range(len(train_data_tmpt)), total=len(train_data_tmpt)):
                if self.uniform:
                    point_set = farthest_point_sample(train_data_tmpt[index], self.npoints)
                else:
                    point_set = train_data_tmpt[index][0:self.npoints, :]
                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                point_set = point_set[:, 0:3]
                self.train_data[index] = point_set
                self.train_targets[index] = train_targets_tmpt[index]

            with open(self.train_save_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_targets], f)

            for index in tqdm(range(len(test_data_tmpt)), total=len(test_data_tmpt)):
                if self.uniform:
                    point_set = farthest_point_sample(test_data_tmpt[index], self.npoints)
                else:
                    point_set = test_data_tmpt[index][0:self.npoints, :]
                point_set[:, 0:3] = self.pc_normalize(point_set[:, 0:3])
                point_set = point_set[:, 0:3]
                self.test_data[index] = point_set
                self.test_targets[index] = test_targets_tmpt[index]

            with open(self.test_save_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_targets], f)

        else:
            print('Load processed data')
            with open(self.train_save_path, 'rb') as f:
                self.train_data, self.train_targets = pickle.load(f)
            with open(self.test_save_path, 'rb') as f:
                self.test_data, self.test_targets = pickle.load(f)
        self.train_data = np.stack(self.train_data, axis=0)
        self.test_data = np.stack(self.test_data, axis=0)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)
        self.train_data = self.train_data[:,:, 0:3]
        self.test_data = self.test_data[:, :, 0:3]
        self.train_data = self.train_data.transpose(0, 2, 1)
        self.test_data = self.test_data.transpose(0, 2, 1)
        print('Data Loading Finished..')

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

class M_modelnet40(iData):
    use_path = False

    def download_data(self, config=None):
        print('Load processed data')
        self.root = '/root/autodl-tmp/DataSet/ModelNet40_Multimodals'  # config.DATA_PATH
        traindata_list = os.listdir(os.path.join(self.root, "train", "pcs"))
        testdata_list = os.listdir(os.path.join(self.root, "test", "pcs"))

        self.train_data = [None] * len(traindata_list)
        self.train_data_mviews = [None] * len(traindata_list)
        self.train_targets = [None] * len(traindata_list)

        self.test_data = [None] * len(testdata_list)
        self.test_data_mviews = [None] * len(testdata_list)
        self.test_targets = [None] * len(testdata_list)

        train_path = os.path.join(self.root, "train.dat")
        if not os.path.exists(train_path):#没有保存过一个整体
            for index in tqdm(range(len(traindata_list)), total=len(traindata_list)):
                with open(os.path.join(self.root, "train", "pcs", traindata_list[index]), 'rb') as f:
                    self.train_data[index], self.train_targets[index] = pickle.load(f)

                pngs = [None] * 10
                for item in range(10):
                    png_name = traindata_list[index].replace("_1024dpts.dat", '_%dview.png' % (item))
                    png_path = os.path.join(self.root, "train", "imgs", png_name)
                    if os.path.exists(png_path):
                        pngs[item] = Image.open(png_path).convert('RGB')
                    else:
                        print(png_path + " not exist")
                self.train_data_mviews[index] = pngs

            self.train_data = np.stack(self.train_data, axis=0)
            # self.train_data_mviews = np.stack(self.train_data_mviews, axis=0)
            self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)

            with open(train_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_data_mviews, self.train_targets], f)

        else:#保存过一个整体
            with open(train_path, 'rb') as f:
                self.train_data, self.train_data_mviews, self.train_targets = pickle.load(f)

        test_path = os.path.join(self.root, "test.dat")
        if not os.path.exists(test_path):  # 没有保存过一个整体
            for index in tqdm(range(len(testdata_list)), total=len(testdata_list)):
                with open(os.path.join(self.root, "test", "pcs", testdata_list[index]), 'rb') as f:
                    self.test_data[index], self.test_targets[index] = pickle.load(f)
                pngs=[None] *10
                for item in range(10):
                    png_name = testdata_list[index].replace("_1024dpts.dat", '_%dview.png' % (item))
                    pngs[item] = Image.open(os.path.join(self.root, "test", "imgs", png_name)).convert('RGB')
                self.test_data_mviews[index] = pngs

            self.test_data = np.stack(self.test_data, axis=0)
            # self.test_data_mviews = np.stack(self.test_data_mviews, axis=0)
            self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)

            with open(test_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_data_mviews, self.test_targets], f)

        else:#保存过一个整体
            with open(test_path, 'rb') as f:
                self.test_data, self.test_data_mviews, self.test_targets = pickle.load(f)

    def download_pretraindata(self, config=None):
        print('Load processed data')
        # conversion_dict = {4: 0, 20: 1, 22: 2, 25: 3}
        self.root = '/root/autodl-tmp/DataSet/ModelNet40_Multimodals_RandMask'  # config.DATA_PATH

        self.train_data = os.listdir(os.path.join(self.root, "train", "pcs"))
        self.train_data_mviews = []
        self.train_targets = []

        self.test_data = None
        self.test_data_mviews = None
        self.test_targets = None

        for index in tqdm(range(len(self.train_data)), total=len(self.train_data)):
            pngs = [None] * 10
            for item in range(10):
                png_name = self.train_data[index].replace("_1024dpts.dat", '_%dview.png' % (item))
                pngs[item] = os.path.join(self.root, "train", "imgs", png_name)
            self.train_data_mviews.append(pngs)
            self.train_data[index] = os.path.join(self.root, "train", "pcs", self.train_data[index])
            with open(self.train_data[index], 'rb') as f:
                _, label = pickle.load(f)
            # self.train_targets.append(conversion_dict[label])
            self.train_targets.append(label)

        self.train_data = np.array(self.train_data)
        self.train_data_mviews = np.array(self.train_data_mviews)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)

        self.test_data, self.test_data_mviews, self.test_targets = self.train_data, self.train_data_mviews, self.train_targets

class M_shapenet55(iData):
    use_path = True

    def download_data(self, config=None):
        print('Load processed data')
        self.root = '/root/autodl-tmp/DataSet/ShapeNet55'
        self.root_Multimodals = '/root/autodl-tmp/DataSet/ShapeNet55_Multimodals'
        train_data_list_file = os.path.join(self.root, 'train.txt')
        test_data_list_file = os.path.join(self.root, 'test.txt')

        with open(train_data_list_file, 'r') as f:
            lines = f.readlines()

        self.train_data = []
        self.train_data_mviews = []
        self.train_targets = []

        with open(test_data_list_file, 'r') as f:
            test_lines = f.readlines()

        self.test_data = []
        self.test_data_mviews = []
        self.test_targets = []
        self.process_data = False
        self.nb_points = 1024

        # lines = test_lines + lines
        self.file_list = []
        cat_labels = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5,
                      '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11,
                      '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16,
                      '02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22,
                      '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28,
                      '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33,
                      '03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39,
                      '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45,
                      '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50,
                      '04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}

        check_list = ['03001627-udf068a6b', '03001627-u6028f63e', '03001627-uca24feec', '04379243-', '02747177-',
                      '03001627-u481ebf18', '03001627-u45c7b89f', '03001627-ub5d972a1', '03001627-u1e22cc04',
                      '03001627-ue639c33f']

        for index, line in enumerate(lines):
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            if taxonomy_id + '-' + model_id not in check_list:
                line = line.strip()
                data_path = os.path.join(self.root, 'shapenet_pc', line)
                if self.process_data:
                    data= np.load(data_path)
                    indices = np.random.choice(data.shape[0], self.nb_points, replace=False)
                    np.save(data_path, data[indices])
                self.train_data.append(data_path)
                png_floder = taxonomy_id + '_' + model_id
                self.train_data_mviews.append(os.path.join(self.root_Multimodals, 'rendering', png_floder))
                self.train_targets.append(cat_labels[taxonomy_id])

        self.train_data = np.array(self.train_data)
        self.train_data_mviews = np.array(self.train_data_mviews)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)

        for index, line in enumerate(test_lines):
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            if taxonomy_id + '-' + model_id not in check_list:
                line = line.strip()
                data_path = os.path.join(self.root, 'shapenet_pc', line)
                if self.process_data:
                    data = np.load(data_path)
                    indices = np.random.choice(data.shape[0], self.nb_points, replace=False)
                    np.save(data_path, data[indices])
                self.test_data.append(data_path)
                png_floder = taxonomy_id + '_' + model_id
                self.test_data_mviews.append(os.path.join(self.root_Multimodals, 'rendering', png_floder))
                self.test_targets.append(cat_labels[taxonomy_id])

        self.test_data = np.array(self.test_data)
        self.test_data_mviews = np.array(self.test_data_mviews)
        self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)

    def download_pretraindata(self, config=None):
        print('Load processed data')
        conversion_dict = {4: 0, 11: 1, 20: 2, 25: 3, 30: 4, 34: 5}
        self.root = '/root/autodl-tmp/DataSet/ShapeNet55_Multimodals_RandMask_Seed1993_Task1'  # config.DATA_PATH

        self.train_data = os.listdir(os.path.join(self.root, "train", "pcs"))
        self.train_data2 = os.listdir(os.path.join(self.root, "val", "pcs"))
        self.train_data.extend(self.train_data2)
        self.train_data_mviews = []
        self.train_targets = []

        self.test_data = None
        self.test_data_mviews = None
        self.test_targets = None

        for index in tqdm(range(len(self.train_data)), total=len(self.train_data)):
            pngs = [None] * 10
            for item in range(10):
                png_name = self.train_data[index].replace("_1024dpts.dat", '_%dview.png' % (item))
                pngs[item] = os.path.join(self.root, "train", "imgs", png_name)
            self.train_data_mviews.append(pngs)
            self.train_data[index] = os.path.join(self.root, "train", "pcs", self.train_data[index])
            with open(self.train_data[index], 'rb') as f:
                _, label = pickle.load(f)
            self.train_targets.append(conversion_dict[label])
            # self.train_targets.append(label)

        self.train_data = np.array(self.train_data)
        self.train_data_mviews = np.array(self.train_data_mviews)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)

        self.test_data, self.test_data_mviews, self.test_targets = self.train_data, self.train_data_mviews, self.train_targets


class M_scanobject(iData):
    use_path = False
    def download_data(self, config=None):
        print('Load processed data')
        self.root = '/root/autodl-tmp/DataSet/ScanObjectNN_Multimodals'  # config.DATA_PATH
        self.nb_points = 1024
        traindata_list = os.listdir(os.path.join(self.root, "train", "pcs"))
        testdata_list = os.listdir(os.path.join(self.root, "test", "pcs"))

        self.train_data = [None] * len(traindata_list)
        self.train_data_mviews = [None] * len(traindata_list)
        self.train_targets = [None] * len(traindata_list)

        self.test_data = [None] * len(testdata_list)
        self.test_data_mviews = [None] * len(testdata_list)
        self.test_targets = [None] * len(testdata_list)

        train_path = os.path.join(self.root, "train.dat")
        if not os.path.exists(train_path):#没有保存过一个整体
            for index in tqdm(range(len(traindata_list)), total=len(traindata_list)):
                with open(os.path.join(self.root, "train", "pcs", traindata_list[index]), 'rb') as f:
                    train_data_tempt, train_targets_tempt = pickle.load(f)
                    train_data_tempt = self.cubetopcs(train_data_tempt)
                    if len(train_data_tempt)<200:
                        continue
                    else:
                        indices = np.random.choice(train_data_tempt.shape[0], self.nb_points, replace=True)
                        self.train_data[index] = train_data_tempt[indices]
                        self.train_targets[index] = train_targets_tempt

                pngs = [None] * 10
                for item in range(10):
                    png_name = traindata_list[index].replace("_cubes.dat", '_%dview.png' % (item))
                    png_path = os.path.join(self.root, "train", "imgs", png_name)
                    if os.path.exists(png_path):
                        pngs[item] = Image.open(png_path).convert('RGB')
                    else:
                        print(png_path + " not exist")
                self.train_data_mviews[index] = pngs

            self.train_data = [data for data in self.train_data if data is not None]
            self.train_data_mviews = [data for data in self.train_data_mviews if data is not None]
            self.train_targets = [data for data in self.train_targets if data is not None]

            self.train_data = np.stack(self.train_data, axis=0)
            # self.train_data_mviews = np.stack(self.train_data_mviews, axis=0)
            self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)

            with open(train_path, 'wb') as f:
                pickle.dump([self.train_data, self.train_data_mviews, self.train_targets], f)

        else:#保存过一个整体
            with open(train_path, 'rb') as f:
                self.train_data, self.train_data_mviews, self.train_targets = pickle.load(f)

        test_path = os.path.join(self.root, "test.dat")
        if not os.path.exists(test_path):  # 没有保存过一个整体
            for index in tqdm(range(len(testdata_list)), total=len(testdata_list)):
                with open(os.path.join(self.root, "test", "pcs", testdata_list[index]), 'rb') as f:
                    # self.test_data[index], self.test_targets[index] = pickle.load(f)
                    test_data_tempt, test_targets_tempt = pickle.load(f)
                    test_data_tempt = self.cubetopcs(test_data_tempt)
                    if len(test_data_tempt) < 200:
                        continue
                    else:
                        indices = np.random.choice(test_data_tempt.shape[0], self.nb_points, replace=True)
                        self.test_data[index] = test_data_tempt[indices]
                        self.test_targets[index] = test_targets_tempt
                pngs=[None] *10
                for item in range(10):
                    png_name = testdata_list[index].replace("_cubes.dat", '_%dview.png' % (item))
                    pngs[item] = Image.open(os.path.join(self.root, "test", "imgs", png_name)).convert('RGB')
                self.test_data_mviews[index] = pngs

            self.test_data = [data for data in self.test_data if data is not None]
            self.test_data_mviews = [data for data in self.test_data_mviews if data is not None]
            self.test_targets = [data for data in self.test_targets if data is not None]

            self.test_data = np.stack(self.test_data, axis=0)
            # self.test_data_mviews = np.stack(self.test_data_mviews, axis=0)
            self.test_targets = np.stack(self.test_targets, axis=0).reshape(-1)

            with open(test_path, 'wb') as f:
                pickle.dump([self.test_data, self.test_data_mviews, self.test_targets], f)

        else:#保存过一个整体
            with open(test_path, 'rb') as f:
                self.test_data, self.test_data_mviews, self.test_targets = pickle.load(f)

    def download_pretraindata(self, config=None):
        print('Load processed data')
        # conversion_dict = {4: 0, 20: 1, 22: 2, 25: 3}
        self.root = '/root/autodl-tmp/DataSet/ScanObjectNN_Multimodals_RandMask'  # config.DATA_PATH

        self.train_data = os.listdir(os.path.join(self.root, "train", "pcs"))
        self.train_data_mviews = []
        self.train_targets = []

        self.test_data = None
        self.test_data_mviews = None
        self.test_targets = None

        for index in tqdm(range(len(self.train_data)), total=len(self.train_data)):
            pngs = [None] * 10
            for item in range(10):
                png_name = self.train_data[index].replace(".dat", '_%dview.png' % (item))
                pngs[item] = os.path.join(self.root, "train", "imgs", png_name)
            self.train_data_mviews.append(pngs)
            self.train_data[index] = os.path.join(self.root, "train", "pcs", self.train_data[index])
            with open(self.train_data[index], 'rb') as f:
                _, label = pickle.load(f)
            # self.train_targets.append(conversion_dict[label])
            self.train_targets.append(label)

        self.train_data = np.array(self.train_data)
        self.train_data_mviews = np.array(self.train_data_mviews)
        self.train_targets = np.stack(self.train_targets, axis=0).reshape(-1)

        self.test_data, self.test_data_mviews, self.test_targets = self.train_data, self.train_data_mviews, self.train_targets

    def cubetopcs(self, cube):
        point_cloud = []
        x_len = len(cube)
        y_len = len(cube[0])
        z_len = len(cube[0][0])
        # 遍历 14x14x14 的体素数据
        for x in range(x_len):
            for y in range(y_len):
                for z in range(z_len):
                    # 获取当前体素的值
                    value = cube[x, y, z]
                    # 检查体素是否有效（例如，取决于值的某种条件）
                    if np.any(value > 0):  # 假设我们只关注值大于零的体素
                        # 将体素的坐标和值添加到点云中
                        point_cloud.append([x, y, z, *value])
        point_cloud = np.array(point_cloud)[:,:3]
        return  point_cloud