from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

def _batch_mid_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 1]
	hard_p_indice = positive_indices[:, 1]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=False, mid_hard=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
		self.mid_hard = mid_hard

	def forward(self, emb, label, emb_=None):
		if emb_ is None:
			mat_dist = euclidean_dist(emb, emb)
			# mat_dist = cosine_dist(emb, emb)
			assert mat_dist.size(0) == mat_dist.size(1)
			N = mat_dist.size(0)
			mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
			if self.mid_hard:
				dist_ap, dist_an = _batch_mid_hard(mat_dist, mat_sim)
			else:
				dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
			assert dist_an.size(0)==dist_ap.size(0)
			y = torch.ones_like(dist_ap)
			loss = self.margin_loss(dist_an, dist_ap, y)
			prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
			return loss, prec
		else:
			mat_dist = euclidean_dist(emb, emb_)
			N = mat_dist.size(0)
			mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
			dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
			y = torch.ones_like(dist_ap)
			loss = self.margin_loss(dist_an, dist_ap, y)
			return loss



class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=False,mid_hard=False):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.mid_hard = mid_hard

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
		if self.mid_hard:
			dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
		else:
			dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = (- triple_dist_ref * triple_dist).mean(0).sum()
		return loss

class SoftTripletLoss_weight(nn.Module):

	def __init__(self, margin=None, normalize_feature=False,mid_hard=False):
		super(SoftTripletLoss_weight, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.mid_hard = mid_hard

	def forward(self, emb1, emb2, label, weights):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
		if self.mid_hard:
			dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
		else:
			dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = torch.sum((- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]) * weights)
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = torch.sum((- torch.sum((triple_dist_ref * triple_dist),dim=1)).view(-1) * weights)
		return loss
	

class RankingLoss:

	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

def tensor_cosine_dist(x, y):
	'''
	compute cosine distance between two matrix x and y
	with size (n1, d) and (n2, d) and type torch.tensor
	return a matrix (n1, n2)
	'''

	x = F.normalize(x, dim=1)
	y = F.normalize(y, dim=1)
	return torch.matmul(x, y.transpose(0,1))


def tensor_euclidean_dist(x, y):
	"""
	compute euclidean distance between two matrix x and y
	with size (n1, d) and (n2, d) and type torch.tensor
	return a matrix (n1, n2)
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(mat1=x.float(), mat2=y.float().t(), beta=1, alpha=-2)
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist
class PlasticityLoss(RankingLoss):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, metric, if_l2='euclidean'):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param bh: batch hard
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.metric = metric
		self.if_l2 = if_l2

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		if self.metric == 'cosine':
			mat_dist = tensor_cosine_dist(emb1, emb2)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			mat_dist = tensor_cosine_dist(emb1, emb3)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			margin_label = -torch.ones_like(hard_p)

		elif self.metric == 'euclidean':
			if self.if_l2:
				emb1 = F.normalize(emb1)
				emb2 = F.normalize(emb2)
			mat_dist = tensor_euclidean_dist(emb1, emb2)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			mat_dist = tensor_euclidean_dist(emb1, emb3)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			margin_label = torch.ones_like(hard_p)

		return self.margin_loss(hard_n, hard_p, margin_label)