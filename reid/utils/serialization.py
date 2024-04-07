from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def copy_state_dict_dsbn(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        index = ['bns.0', 'bns.1', 'bns.2', 'bns.3']
        if isinstance(param, Parameter):
            param = param.data
        if 'bns.0' in name:
            for ind in index:
                new_name = name.replace('bns.0', ind)
                tgt_state[new_name].copy_(param)
                copied_names.add(new_name)
        else:
            if name not in tgt_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            if param.size() != tgt_state[name].size():
                print('mismatch:', name, param.size(), tgt_state[name].size())
                continue
            tgt_state[name].copy_(param)
            copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()

    copied_names = set()
    for name, param in state_dict.items():

        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)
    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def copy_state_dict_save_prompt(state_dict, model, strip=None,index=0):
    tgt_state = model.state_dict()

    copied_names = set()
    for name, param in state_dict.items():
        
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        #sprint(name)
        tgt_state[name].copy_(param)
        copied_names.add(name)
        #print(name,param.numpy().mean())
        if name == 'module.base.general_prompt':
            b = param.numpy().tolist()
        if name == 'module.base.pool.key_list':
            c = param.numpy().tolist() 
        if name == 'module.base.pool.prompt_list':
            a = param.numpy().tolist()

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)
    
    data = {'prompt_list':a,'general_prompt':b,'key_list0':c}
    with open('comp_p3/data{}.json'.format(index),'w') as f:
        json.dump(data,f)
    #exit(0)
    return model

def copy_state_dict_save_param(state_dict, model, strip=None,index=0):

    tgt_state = model.state_dict()
    data = {}
    copied_names = set()
    for name, param in state_dict.items():
        
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        #sprint(name)
        tgt_state[name].copy_(param)
        copied_names.add(name)
        layer = 11
        #print(name,param.numpy().mean())
        if name == 'module.base.general_prompt':
            data[name]=param.numpy().tolist()
        for num in range(6,12):
            if (str(num) in name)  and ('qkv' in name):
                data[name]=param.numpy().tolist()
                print(name,param.numpy().max(),param.numpy().min(),param.numpy().mean())
        '''
        for num in range(6,12):
            if (str(num) in name) and (layer==num) and ('proj' in name):
                data[name]=param.numpy().tolist()
                print(name)
        '''
    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)
    print('--------------------------')

    #with open('param_ls/l11/data{}_proj.json'.format(index),'w') as f:
    #    json.dump(data,f)
    #exit(0)
    return model
