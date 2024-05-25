import numpy as np
import torch
import json
import os
import dgl
from dgl.nn.pytorch import GraphConv
import time
import torch.nn.functional as F

from utils1 import *
from config import *
from utils2 import get_link


def my_check_align1(pred, ground_truth, result_file=None):
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()

    g_list = list(g_map.keys())
    ind = (-pred).argsort(axis=1)[:, :10]

    a1, a5, a10, mrr = 0, 0, 0, 0
    for i, node in enumerate(g_list):
        for j in range(10):
            if ind[node, j].item() == g_map[node]:
                if j < 1:
                    a1 += 1
                if j < 5:
                    a5 += 1
                if j < 10:
                    a10 += 1
                mrr += 1.0 / (j + 1)
    a1 /= len(g_list)
    a5 /= len(g_list)
    a10 /= len(g_list)
    mrr = mrr / len(g_list)
    print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% MRR %.4f%%' % (a1 * 100, a5 * 100, a10*100, mrr))
    return a1,a5,a10,mrr


def gw_torch(flag, X2, cost_s, cost_t, p_s=None, p_t=None, trans=None, beta=0.01, error_bound=1e-6, outer_iter=1000, inner_iter=10):

    device = cost_s.device

    gt = torch.tensor(np.array(list(range(cost_s.shape[0]))), device=device)
    if p_s is None:
        p_s = torch.ones([cost_s.shape[0],1], device=device)/cost_s.shape[0]
    if p_t is None:
        p_t = torch.ones([cost_t.shape[0],1], device=device)/cost_t.shape[0]
    if trans is None:
        trans = p_s @ p_t.T
    obj_list = []
    if X2 != None:
        ground_truth = torch.cat(
            [torch.tensor(list(range(4500, X2.shape[0]))).unsqueeze(0),
             torch.tensor(list(range(4500, X2.shape[0]))).unsqueeze(0)], 0)
    for oi in range(outer_iter):
        time_st = time.time()
        cost = - 10 * (cost_s @ trans @ cost_t.T)

        kernel = torch.exp(-cost / beta) * trans

        a = torch.ones_like(p_s)/p_s.shape[0]

        for ii in range(inner_iter):
            b = p_t / (kernel.T@a)
            a_new = p_s / (kernel@b)
            relative_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
            a = a_new
            if relative_error < 1e-6:
                break
        trans = (a @ b.T) * kernel



        if oi % 100 == 0 and oi > 2:
            obj = (cost_s ** 2).mean() + (cost_t ** 2).mean() -torch.trace(cost_s @ trans @ cost_t @ trans.T)

            print(oi, obj)
            if len(obj_list) > 0:
                print('obj gap: ', torch.abs(obj-obj_list[-1])/obj_list[-1])
                if torch.abs(obj-obj_list[-1])/obj_list[-1] < error_bound:
                    print('iter:{}, smaller than eps'.format(ii))
                    break
            obj_list.append(obj.item())
            if flag:
                print(flag)
                a1, a5, a10, mrr = my_check_align1((trans + X2) / 2, ground_truth)
                time_ed = time.time()
                with open('result.txt', 'a+') as f:
                    f.write(
                        'Epoch: {} H@1 {:.3f} H@5 {:.3f} H@10 {:.3f} MRR {:.4f} Time {:.2f}\n'.format(oi,
                            a1, a5, a10, mrr, time_ed - time_st))

    return trans




def SLOTAlign(flag, As, Bs, X,  step_size=1, gw_beta=0.01, joint_epoch=100, gw_epoch=2000, X2=None):
    # Add this code to handle 0-dim input vector
    layers = As.shape[2]
    alpha0 = np.ones(layers).astype(np.float32)/layers
    beta0 = np.ones(layers).astype(np.float32)/layers
    Adim, Bdim = As.shape[0], Bs.shape[0]
    a = torch.ones([Adim,1]).cuda()/Adim
    b = torch.ones([Bdim,1]).cuda()/Bdim
    # a = torch.ones([Adim, 1]) / Adim
    # b = torch.ones([Bdim, 1]) / Bdim
    if X2 != None:
        ground_truth = torch.cat(
            [torch.tensor(list(range(4500, X2.shape[0]))).unsqueeze(0),
             torch.tensor(list(range(4500, X2.shape[0]))).unsqueeze(0)], 0)
    for ii in range(joint_epoch):
        time_st = time.time()
        alpha = torch.autograd.Variable(torch.tensor(alpha0)).cuda()
        # alpha = torch.autograd.Variable(torch.tensor(alpha0))
        alpha.requires_grad = True
        beta = torch.autograd.Variable(torch.tensor(beta0)).cuda()
        # beta = torch.autograd.Variable(torch.tensor(beta0))
        beta.requires_grad = True
        A = (As * alpha).sum(2)
        B = (Bs * beta).sum(2)
        objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ X @ B @ X.T)
        alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
        alpha = alpha - step_size * alpha_grad
        alpha0 = alpha.detach().cpu().numpy()
        alpha0 = euclidean_proj_simplex(alpha0)
        beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
        beta = beta - step_size * beta_grad
        beta0 = beta.detach().cpu().numpy()
        beta0 = euclidean_proj_simplex(beta0)

        X = gw_torch(flag, X2, A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=gw_beta, outer_iter=1,
                     inner_iter=50).clone().detach()
        if ii == 99:
            X = gw_torch(flag, X2, A.clone().detach(), B.clone().detach(), a, b, X, beta=0.01, outer_iter=gw_epoch, inner_iter=2)


    return X


def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    # rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    rho = np.sum(u * np.arange(1, n+1) > (cssv - s))-1
    rho = rho if rho > 0 else 0

    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def NeuralSinkhorn(cost, p_s=None, p_t=None, trans=None, beta=0.1, outer_iter=20):
    if p_s is None:
        p_s = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    if p_t is None:
        p_t = torch.ones([cost.shape[1],1],device=cost.device)/cost.shape[1]
    if trans is None:
        trans = p_s @ p_t.T
    a = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    cost_new = torch.exp(-cost / beta)

    p_s_initial = p_s.clone()
    p_t_initial = p_t.clone()

    for oi in range(outer_iter):
        kernel = cost_new * trans
        # b = p_t / (kernel.T@a)
        # a = p_s / (kernel@b)
        b = p_t_initial / (kernel.T @ a)
        a = p_s_initial / (kernel @ b)
        trans = (a @ b.T) * kernel
    return trans


def run_DATASET(filename, translate=True, layers=2, gw_beta=0.001, gw_epoch=1000, step_size=1):
    train_pair, dev_pair, adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features = load_data(filename, train_ratio=train_ratio, unsup=unsup)
    adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
    rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
    time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
    test_pair = get_link(filename)
    test_pair = np.array(test_pair)
    feature = torch.load('data/ICEWS05-15/features.pt').to('cpu')
    # print(feature.shape)
    # size = (19054, 768)
    time_features = torch.load('embedding_ICE/time_featuers.pt').to('cpu')
    rel_features = torch.load('embedding_ICE/out_feature_rel.pt').to('cpu')
    ent_features = torch.load('embedding_ICE/out_feature_ent.pt').to('cpu')

    # time_features = torch.concat([time_features, rel_features], dim=1)
    # time_features = torch.concat([time_features, ent_features], dim=1)
    time_features = torch.cat([time_features, rel_features], dim=1)
    time_features = torch.cat([time_features, ent_features], dim=1)

    node_size = adj_matrix.shape[0]
    time_size = time_features.shape[1]
    rel_size = rel_features.shape[1]

    # feature = torch.concat([feature, time_features], dim=1)
    feature = torch.cat([feature, time_features], dim=1)

    adj_mat = torch.sparse_coo_tensor(adj_matrix.T, np.ones(len(adj_matrix)), (node_size, node_size)).float()

    idx = adj_mat.coalesce().indices()
    G = dgl.graph((idx[0], idx[1]))
    G = dgl.add_self_loop(G)
    adj_mat_dense = G.adj().to_dense()
    Aadj = adj_mat_dense[test_pair[:,0]][:,test_pair[:,0]]
    Badj = adj_mat_dense[test_pair[:,1]][:,test_pair[:,1]]
    #
    Aadj/=Aadj.max()
    Badj/=Badj.max()
    time_st = time.time()
    new_feature = feature / (feature.norm(dim=1)[:, None]+1e-16)
    features = [new_feature]

    new_time_features = time_features / (time_features.norm(dim=1)[:, None] + 1e-16)
    time_features = [new_time_features]

    conv = GraphConv(0, 0, norm='both', weight=False, bias=False)
    for i in range(layers):
        new_feature = conv(G,features[-1])
        new_feature = new_feature / (new_feature.norm(dim=1)[:, None]+1e-16)
        features.append(new_feature)

        new_time_features = conv(G, time_features[-1])
        new_time_features = new_time_features /(new_time_features.norm(dim=1)[:, None]+1e-16)
        time_features.append(new_time_features)
    Asims, Bsims = [Aadj], [Badj]

    col = 0
    for feature in features:
        Asim = feature[test_pair[:,0]].mm(feature[test_pair[:,0]].T)
        Asim2 = time_features[col][test_pair[:,0]].mm(time_features[col][test_pair[:,0]].T)

        u, s, vt = torch.linalg.svd(Asim)
        u2, s2, vt2 = torch.linalg.svd(Asim2)
        k = 20
        u_k = u[:, :k]
        u2_k = u2[:, :k]
        Asim = u_k @ u2_k.T
        Asims.append(Asim)

        Bsim = feature[test_pair[:,1]].mm(feature[test_pair[:,1]].T)
        Bsim2 = time_features[col][test_pair[:,1]].mm(time_features[col][test_pair[:,1]].T)

        u, s, vt = torch.linalg.svd(Bsim)
        u2, s2, vt2 = torch.linalg.svd(Bsim2)
        k = 20
        u_k = u[:, :k]
        u2_k = u2[:, :k]
        Bsim = u_k @ u2_k.T
        Bsims.append(Bsim)
        col = col + 1
    sim1 = features[0][test_pair[:,0]].mm(features[0][test_pair[:,1]].T)
    sim2 = time_features[0][test_pair[:,0]].mm(time_features[0][test_pair[:,1]].T)
    initX=NeuralSinkhorn(1-sim1.float().cuda())
    initX2=NeuralSinkhorn(1-sim2.float().cuda())
    As = torch.stack(Asims,dim=2).cuda()
    Bs = torch.stack(Bsims,dim=2).cuda()
    # initX = NeuralSinkhorn(1 - sim1.float())
    # As = torch.stack(Asims, dim=2)
    # Bs = torch.stack(Bsims, dim=2)
    # print(initX)
    flag = False
    X2 = SLOTAlign(flag, As, Bs, initX2, step_size=step_size, gw_beta=gw_beta, gw_epoch=gw_epoch, X2=None)
    flag = True
    X = SLOTAlign(flag, As, Bs, initX, step_size=step_size, gw_beta=gw_beta, gw_epoch=gw_epoch, X2=X2)



run_DATASET('data/ICEWS05-15/', translate=False, layers=2, gw_beta=10, gw_epoch=6000, step_size=1)
# run_DATASET('data/YAGO-WIKI50K/', translate=False, layers=2, gw_beta=10, gw_epoch=6000, step_size=1)
