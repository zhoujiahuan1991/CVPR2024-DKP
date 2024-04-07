import copy

import torch.nn as nn
import torchvision.models as models

import torchvision
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)


class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-4)
        return out


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model='resnet50', out_dim=2048, n_sampling=2, pool_len=8, normal_feature=True,
                 num_classes=500, uncertainty=False):
        super(ResNetSimCLR, self).__init__()   
       
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=True)}
        self.resnet = self._get_basemodel(base_model)
        self.base = nn.Sequential(*list(self.resnet.children())[:-3])
        dim_mlp = self.resnet.fc.in_features//2
        self.linear_mean =nn.Linear(dim_mlp, out_dim)
        self.linear_var = nn.Linear(dim_mlp, out_dim)
        self.pool_len = 8
        self.conv_var =  nn.Conv2d(dim_mlp, dim_mlp, kernel_size=(pool_len,pool_len),bias=False)

        self.n_sampling = n_sampling
        self.n_samples = torch.Size(np.array([n_sampling, ]))
        self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.l2norm_mean, self.l2norm_var, self.l2norm_sample = Normalize(2, 1), Normalize(2, 1), Normalize(2, 2)

        print('using resnet50 as a backbone')
        '''xkl add'''
        print("##########normalize matchiing feature:", normal_feature)
        self.normal_feature = normal_feature
        self.uncertainty = uncertainty

        self.bottleneck = nn.BatchNorm2d(out_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.classifier = nn.Linear(out_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.relu = nn.ReLU()    

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x, training_phase=None, fkd=False):
        BS = x.size(0)
        
        out = self.base(x)  # former 3 blockes of resnet 50
        out_mean = self.pooling_layer(out)  # global pooling
        out_mean = out_mean.view(out_mean.size(0), -1)  # B x 1024
        out_mean = self.linear_mean(out_mean)  # Bx2048
        # out_mean = self.l2norm_mean(out_mean)  # L2norm

        out_var = self.conv_var(out)  # conv layer
        out_var = self.pooling_layer(out_var)  # pooling
        out_var += 1e-4
        out_var = out_var.view(out_var.size(0), -1)  # Bx1024
        out_var = self.linear_var(out_var)  # Bx2049

        out_mean=self.l2norm_mean(out_mean)
        
        var_choice = 'L2'
        if var_choice == 'L2':
            out_var = self.l2norm_var(out_var)
            out_var = self.relu(out_var)+ 1e-4
        elif var_choice == 'softmax':
            out_var = F.softmax(out_var, dim=1)# Bx2049
            out_var = out_var.clone()  # gradient computation error would occur without this line
        elif var_choice=='log':
            out_var=torch.exp(0.5 * out_var)
           
        if self.uncertainty:
            BS,D=out_mean.size()                
            tdist = MultivariateNormal(loc=out_mean, scale_tril=torch.diag_embed(out_var))
            samples = tdist.rsample(self.n_samples)  # (n_samples, batch_size, out_dim)

            # if self.normal_feature:
            samples = self.l2norm_sample(samples)

            merge_feat = torch.cat((out_mean.unsqueeze(0), samples), dim=0)  # (n_samples+1,batchsize, out_dim)
            merge_feat = merge_feat.resize(merge_feat.size(0) * merge_feat.size(1),
                                           merge_feat.size(-1))  # ((n_samples+1)*batchsize, out_dim)
            bn_feat = self.bottleneck(
                merge_feat.unsqueeze(-1).unsqueeze(-1))  # [(n_samples+1)*batchsize, out_dim, 1, 1]
            cls_outputs = self.classifier(bn_feat[..., 0, 0])  # [(n_samples+1)*batchsize, num_classes]

            merge_feat = merge_feat.resize(self.n_sampling + 1, BS,
                                           merge_feat.size(-1))  # (n_samples+1,batchsize, out_dim)
            cls_outputs = cls_outputs.resize(self.n_sampling + 1, BS,
                                             cls_outputs.size(-1))  # (n_samples+1,batchsize, num_classes)
        else:
            bn_feat = self.bottleneck(out_mean.unsqueeze(-1).unsqueeze(-1))  # [batch_size, 2048, 1, 1]
            cls_outputs = self.classifier(bn_feat[..., 0, 0])  # [batch_size, num_classes]
            cls_outputs = cls_outputs.unsqueeze(0)  # [1, batch_size, num_classes]
            merge_feat = out_mean.unsqueeze(0)  # [1, batch_size, 2048]
        if fkd:  # return all features
            return out_mean, merge_feat.permute(1, 0, 2), cls_outputs.permute(1, 0, 2), out_var, out
        if self.training:# return all features
            return out_mean, merge_feat.permute(1, 0, 2), cls_outputs.permute(1, 0, 2), out_var, out
        else:  # return mean for evalutaion
            return out_mean[:BS]


if __name__ == '__main__':
    m = ResNetSimCLR(uncertainty=True)
    m(torch.zeros(10, 3, 256, 128))
