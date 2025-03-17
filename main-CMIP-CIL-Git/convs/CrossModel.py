# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import resnet50, resnet18

class Adapter(nn.Module):
    def __init__(self,
                 dropout=0.1,
                 init_option="lora"):
        super().__init__()
        self.n_embd = 512 # config.d_model if d_model is None else d_model
        self.down_size = 256 # config.attn_bn if bottleneck is None else bottleneck
        self.n_embd2 = 512

        #_before
        self.scale = nn.Parameter(torch.ones(1))
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd2)
        self.dropout = 0.2
        self.bn = nn.BatchNorm1d(512)
        '''
        self.scale2 = nn.Parameter(torch.ones(1))
        self.down_proj2 = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func2 = nn.ReLU()
        self.up_proj2 = nn.Linear(self.down_size, self.n_embd)
        # self.bn = nn.BatchNorm1d(256)
        
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        '''

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        # up = up * self.scale

        '''
        down = self.down_proj2(up)
        down = self.non_linear_func2(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj2(down)
        up = up * self.scale2
        '''
        if add_residual:
            output = up + residual
        else:
            output = up
        # output = self.bn(output)
        return output


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    ###design for scanobject#####
    # num_points = pairwise_distance.size(-1)
    # k = min(k, num_points)
    #############################

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda:0')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class PCS_Mviews_CrossModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("I'm using CrossModel with adapters.")
        self.args = args
        self.point_model = DGCNN(self.args).cuda()
        self.img_model = ResNet(resnet50(), feat_dim=2048).cuda()
        self.out_dim = 512
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(512)

        self.adapter_list = []
        if self.args["pretrain"] == False:
            self.get_new_adapter()

    def get_new_adapter(self):
        self.cur_adapter = None
        self.cur_adapter = Adapter().cuda()
        self.cur_adapter.requires_grad_(True)

    def add_adapter_to_list(self):
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def forword_basic(self, x, x2, x_mviews, layerout = False):
        x = torch.cat((x, x2))
        x = x.transpose(2, 1).contiguous()
        _, point_feats, _ = self.point_model(x)
        point_feats1 = point_feats[:int(len(x) / 2), :]
        point_feats2 = point_feats[int(len(x) / 2):, :]
        layers_feats , img_feats = self.img_model(x_mviews, layerout)
        img_feats = img_feats.view(-1, 10, img_feats.size()[-1])
        img_feats = torch.mean(img_feats, dim=1)

        # point_feats1 = self.bn1(point_feats1)
        # point_feats2 = self.bn2(point_feats2)
        # img_feats = self.bn3(img_feats)

        return point_feats1, point_feats2, img_feats, layers_feats

    def forward(self, x, x2, x_mviews, layerout = False , test = False):
        assert not torch.isnan(x_mviews).any()
        point_feats1, point_feats2, img_feats, layers_feats = self.forword_basic(x, x2, x_mviews, layerout)

        if self.args["pretrain"] == True:# 预训练
            return point_feats1, point_feats2, img_feats
        else: # 非预训练
            adapt = self.cur_adapter
            point_feats1 = adapt(point_feats1)
            point_feats2 = adapt(point_feats2)
            img_feats = adapt(img_feats)
            return point_feats1, point_feats2, img_feats, layers_feats
            '''
            if not test:
                adapt = self.cur_adapter
                point_feats1 = adapt(point_feats1)
                point_feats2 = adapt(point_feats2)
                img_feats = adapt(img_feats)
                return point_feats1, point_feats2, img_feats
            else:
                point_feats1_list, point_feats2_list, img_feats_list = [], [], []
                for i in range(len(self.adapter_list)):
                    adapt = self.adapter_list[i]
                    point_feats1_list.append(adapt(point_feats1))
                    point_feats2_list.append(adapt(point_feats2))
                    img_feats_list.append(adapt(img_feats))

                adapt = self.cur_adapter
                point_feats1_list.append(adapt(point_feats1))
                point_feats2_list.append(adapt(point_feats2))
                img_feats_list.append(adapt(img_feats))

                point_feats1_output, point_feats2_output, img_feats_output = torch.Tensor().cuda(), torch.Tensor().cuda(), torch.Tensor().cuda(),
                for x in point_feats1_list:
                    point_feats1_output = torch.cat((
                        point_feats1_output,
                        x
                    ), dim=1)

                for x in point_feats2_list:
                    point_feats2_output = torch.cat((
                        point_feats2_output,
                        x
                    ), dim=1)

                for x in img_feats_list:
                    img_feats_output = torch.cat((
                        img_feats_output,
                        x
                    ), dim=1)
                return point_feats1_output, point_feats2_output, img_feats_output
            '''

    def forward_proto(self, x, x2, x_mviews, adapt_index):
        point_feats1, point_feats2, img_feats = self.forword_basic(x, x2, x_mviews)

        i = adapt_index
        if i < len(self.adapter_list):
            adapt = self.adapter_list[i]
        else:
            adapt = self.cur_adapter

        point_feats1 = adapt(point_feats1)
        point_feats2 = adapt(point_feats2)
        img_feats = adapt(img_feats)

        return point_feats1, point_feats2, img_feats
class DGCNN(nn.Module):
    def __init__(self, args, cls=-1):
        super(DGCNN, self).__init__()
        output_channels = 128
        self.args = args
        self.k = args['k']

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args['emb_dims'])

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args['emb_dims'], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        if cls != -1:
            self.linear1 = nn.Linear(args['emb_dims'] * 2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p = 0.5)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p = 0.5)
            self.linear3 = nn.Linear(256, output_channels)

        self.cls = cls

        self.inv_head = nn.Sequential(
            nn.Linear(args['emb_dims'] * 2, args['emb_dims']),
            nn.BatchNorm1d(args['emb_dims']),
            nn.ReLU(inplace=True),
            nn.Linear(args['emb_dims'], 512)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        feat = x
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)

        inv_feat = self.inv_head(feat)

        return x, inv_feat, feat

class ResNet(nn.Module):
    def __init__(self, model, feat_dim=2048):
        super(ResNet, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Identity()

        self.inv_head = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False)
        )

    def forward(self, x, layerout = False):
        if layerout == False:
            if torch.isnan(x).any():
                shiyan = 0
            x_mviews = self.resnet(x)
            assert not torch.isnan(x_mviews).any()
            x_mviews = self.inv_head(x_mviews)
            assert not torch.isnan(x_mviews).any()
            return None, x_mviews
        else:
            # 获取模型的各个层
            layer1 = nn.Sequential(*list(self.resnet.children())[:5])  # 从输入到第一层
            layer2 = nn.Sequential(*list(self.resnet.children())[5])  # 第二层
            layer3 = nn.Sequential(*list(self.resnet.children())[6])  # 第三层
            layer4 = nn.Sequential(*list(self.resnet.children())[7])  # 第四层
            layer5 = nn.Sequential(*list(self.resnet.children())[8:])  # 到输出

            x_1 = layer1(x)
            x_2 = layer2(x_1)
            x_3 = layer3(x_2)
            x_4 = layer4(x_3)
            x_mviews= self.resnet.fc(layer5[0](x_4)[:,:,0,0])
            x_mviews = self.inv_head(x_mviews)  # 最后输出
            return [x_1, x_2, x_3, x_4], x_mviews

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

