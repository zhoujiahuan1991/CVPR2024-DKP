from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn

from reid.loss.loss_uncertrainty import TripletLoss_set
from .utils.meters import AverageMeter
from reid.metric_learning.distance import cosine_similarity

class Trainer(object):
    def __init__(self,args, model, writer=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.writer = writer
        self.uncertainty=True
        if self.uncertainty:
            self.criterion_ce=nn.CrossEntropyLoss()
            self.criterion_triple=TripletLoss_set()

        
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        
        self.AF_weight=args.AF_weight   # anti-forgetting loss
             
        self.n_sampling=args.n_sampling        
       
    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None, proto_type=None        
              ):       
        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        if proto_type is not None:
            proto_type_merge={}
            steps=list(proto_type.keys())
            steps.sort()
            stages=1
            if stages<len(steps):
                steps=steps[-stages:]

            proto_type_merge['mean_features']=torch.cat([proto_type[k]['mean_features'] for k in steps])
            proto_type_merge['labels'] = torch.tensor([proto_type[k]['labels'] for k in steps]).to(proto_type_merge['mean_features'].device)
            proto_type_merge['mean_vars'] = torch.cat([proto_type[k]['mean_vars'] for k in steps])
           
            features_mean=proto_type_merge['mean_features']

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs, targets, cids, domains, = self._parse_data(train_inputs)
            targets += add_num
            # BS*2048, BS*(1+n_sampling)*2048,BS*(1+n_sampling)*num_classes, BS*2048
            s_features, merge_feat, cls_outputs, out_var,feat_final_layer = self.model(s_inputs)

            loss_ce, loss_tp = 0, 0
            if self.uncertainty:
                ###ID loss###
                for s_id in range(1 + self.n_sampling):
                    loss_ce += self.criterion_ce(cls_outputs[:, s_id], targets)
                loss_ce = loss_ce / (1 + self.n_sampling)
                loss_ce=loss_ce*1
                ###set triplet-loss##
                loss_tp = self.criterion_triple(merge_feat, targets)[0]
                loss_tp=loss_tp*1.5            

            
            loss = loss_ce + loss_tp          

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tp.item())


            if training_phase>1:
                divergence=0.                       
                center_feat=merge_feat[:,0]  # obtain the center feature
                Affinity_matrix_new = self.get_normal_affinity(center_feat, self.args.lambda_2) # obtain the affinity matrix
                features_var = proto_type_merge['mean_vars']    # obtain the prototype strandard variances
                noises = torch.randn(features_mean.size()).to(features_mean.device) # generate gaussian noise
                samples = noises * features_var + features_mean # obtain noised sample
                samples = F.normalize(samples, dim=1)   # normalize the sample
                s_features_old = cosine_similarity(center_feat, samples)    # obtain the new-old relation
                s_features_old = F.softmax(s_features_old /self.args.lambda_1, dim=1)  # normalize the relation

                Affinity_matrix_old = self.get_normal_affinity(s_features_old, self.args.lambda_2)  # obtain the affinity matrix under the prototype view
                # divergence += self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets)
                Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
                divergence+=self.KLDivLoss(Affinity_matrix_new_log, Affinity_matrix_old)
                
                loss = loss + divergence * self.AF_weight                               
            
            optimizer.zero_grad()

            loss.backward()
            
                     
            optimizer.step()
           
            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'                     
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              ))
        


    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix




    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains



