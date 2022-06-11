# File   : run.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import re
import os
import yaml
import random
import logging
import shutil
import numpy as np
import torch
import argparse
from otrans.model import End2EndModel, LanguageModel
from otrans.train.scheduler import BuildOptimizer, BuildScheduler
from otrans.train.trainer import Trainer
from otrans.utils import count_parameters
from otrans.data.loader import FeatureLoader
from otrans.train.utils import map_to_cuda
from patch import patch_transformer
import editdistance
import time
from torch.nn import functional as F
try:
    from frob import FactorizedConv, frobdecay
except ImportError:
    print("Failed to import factorization")
try:
    from frob import FactorizedLinear, batch_spectral_init, frobenius_norm, patch_module, non_orthogonality
except ImportError:
    print("Failed to import factorization")

class FactorizedEmbedding(FactorizedLinear):

    def __init__(self, embedding, **kwargs):
        embedding.bias = None
        super().__init__(embedding, **kwargs)
        self.kwargs = {'padding_idx': embedding.padding_idx,
                       'max_norm': embedding.max_norm,
                       'norm_type': embedding.norm_type,
                       'scale_grad_by_freq': embedding.scale_grad_by_freq,
                       'sparse': embedding.sparse}
        self.embedding = embedding
        self.max_positions = embedding.max_positions if hasattr(embedding, 'max_positions') else None
        self.embedding.weight = self.U

    def forward(self, *x):
        return F.linear(self.embedding(*x), self.VT.T)



def build_optimizer(params, no_decay=[]):
    params = [{'params': filter(lambda p: p.requires_grad, params)},
              {'params': filter(lambda p: p.requires_grad, no_decay), 'weight_decay': 0.0}]
    return params


def main(args, params, expdir):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model_type = params['model']['type']
    if model_type[-2:] == 'lm':
        model = LanguageModel[model_type](params['model'])
    else:
        model = End2EndModel[model_type](params['model'])
    if 0.0 < args.rank_scale < 1.0:
        patch_transformer(args, model)
        if args.wd2fd:
            no_decay, skiplist = ['w_1', 'w_2', '.conv1', '.conv2','pointwise_conv1','pointwise_conv2',], []
        else:
            no_decay, skiplist = [], ['w_1', 'w_2', '.conv1', '.conv2']
    else:
        no_decay, skiplist = [], []
    #spectral_init(args, model)
    #no_decay, skiplist = [], []

    if args.wd2fd_quekey:
        no_decay.extend(['_query.weight', '_key.weight'])
    else:
        skiplist.append('quekey')
    if args.wd2fd_outval:
        no_decay.extend(['_value.weight', 'output_perform.weight'])
    else:
        skiplist.append('outval')
    # Count total parameters
    count_parameters(model.named_parameters())

    if args.ngpu >= 1:
        model.cuda()
    logging.info(model)
    optimizer = BuildOptimizer[params['train']['optimizer_type']](
        filter(lambda p: p.requires_grad, model.parameters()), **params['train']['optimizer']    )
    logger.info('[Optimizer] Build a %s optimizer!' % params['train']['optimizer_type'])
    scheduler = BuildScheduler[params['train']['scheduler_type']](optimizer, **params['train']['scheduler'])
    logger.info('[Scheduler] Build a %s scheduler!' % params['train']['scheduler_type'])

    if args.continue_training and args.init_model:
        chkpt = torch.load(args.init_model)
        model.load_model(chkpt)
        logger.info('[Continue Training] Load saved model %s' % args.init_model)

    if args.continue_training and args.init_optim_state:
        ochkpt = torch.load(args.init_optim_state)
        optimizer.load_state_dict(ochkpt['optim'])
        logger.info('[Continue Training] Load saved optimizer state dict!')

        global_step = ochkpt['global_step'] if 'global_step' in ochkpt else args.from_step
        scheduler.global_step = global_step
        scheduler.set_lr()
        logger.info('Set the global step to %d and init lr to %.6f' % (scheduler.global_step, scheduler.lr))

    trainer = Trainer(args,params, model=model, optimizer=optimizer, scheduler=scheduler,skiplist=skiplist, expdir=expdir, ngpu=args.ngpu,
                      parallel_mode=args.parallel_mode, local_rank=args.local_rank, is_debug=args.debug,
                      keep_last_n_chkpt=args.keep_last_n_chkpt, from_epoch=args.from_epoch)

    train_loader = FeatureLoader(params, 'train', ngpu=args.ngpu, mode=args.parallel_mode)
    trainer.train(train_loader=train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='egs/aishell/conf/transformer_baseline.yaml')
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-g', '--gpus', type=str, default='0')
    parser.add_argument('-se', '--seed', type=int, default=1234)
    parser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    parser.add_argument('-r', '--local_rank', type=int, default=0)
    parser.add_argument('-l', '--logging_level', type=str, default='info', choices=['info','debug'])
    parser.add_argument('-lg', '--log_file', type=str, default=None)
    parser.add_argument('-mp', '--mixed_precision', action='store_true', default=False)
    parser.add_argument('-ct', '--continue_training', action='store_true', default=False)
    parser.add_argument('-dir', '--expdir', type=str, default=None)
    parser.add_argument('-im', '--init_model', type=str, default=None)
    parser.add_argument('-ios', '--init_optim_state', type=str, default=None)
    parser.add_argument('-debug', '--debug', action='store_true', default=False)
    parser.add_argument('-knpt', '--keep_last_n_chkpt', type=int, default=30)
    parser.add_argument('-tfs', '--from_step', type=int, default=0)
    parser.add_argument('-tfe', '--from_epoch', type=int, default=0)
    parser.add_argument('-vb', '--verbose', type=int, default=0)
    parser.add_argument('-ol', '--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O1')
    parser.add_argument('--rank-scale', default=1, type=float)
    parser.add_argument('--spectral-quekey', action='store_true')
    parser.add_argument('--spectral-outval', action='store_true')
    parser.add_argument('--spectral', action='store_true')
    parser.add_argument('--frobenius-decay', default=0.0, type=float)
    parser.add_argument('--wd2fd-quekey', action='store_true')
    parser.add_argument('--wd2fd-outval', action='store_true')
    parser.add_argument('--wd2fd', action='store_true')
    parser.add_argument('--square', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-nb', '--nbest', type=int, default=1)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-pn', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m', '--load_model', type=str, default=None)
    parser.add_argument('-lm', '--load_language_model', type=str, default=None)
    parser.add_argument('-ngram', '--ngram_lm', type=str, default=None)
    parser.add_argument('-alpha', '--alpha', type=float, default=0.1)
    parser.add_argument('-beta', '--beta', type=float, default=0.0)
    parser.add_argument('-lmw', '--lm_weight', type=float, default=0.1)
    parser.add_argument('-cw', '--ctc_weight', type=float, default=0.0)
    parser.add_argument('-d', '--decode_set', type=str, default='test')
    parser.add_argument('-ml', '--max_len', type=int, default=60)
    parser.add_argument('-md', '--mode', type=str, default='beam')
    # transducer related
    parser.add_argument('-mt', '--max_tokens_per_chunk', type=int, default=5)
    parser.add_argument('-pf', '--path_fusion', action='store_true', default=False)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-p2w', '--piece2word', action='store_true', default=False)
    parser.add_argument('-resc', '--apply_rescoring', action='store_true', default=False)
    parser.add_argument('-lm_resc', '--apply_lm_rescoring', action='store_true', default=False)
    parser.add_argument('-rw', '--rescore_weight', type=float, default=1.0)
    parser.add_argument('-sba', '--sort_by_avg_score', action='store_true', default=False)
    parser.add_argument('-ns', '--num_sample', type=int, default=1)
    global cmd_args
    cmd_args = parser.parse_args()


    with open('./'+cmd_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if cmd_args.expdir is not None:
        expdir = os.path.join(cmd_args.expdir, params['train']['save_name'])
    else:
        expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    shutil.copy(cmd_args.config, os.path.join(expdir, 'config.yaml'))

    logging_level = {
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    if cmd_args.log_file is not None:
        log_file = cmd_args.log_file
    else:
        log_file = cmd_args.config.split('/')[-1][:-5] + '.log'
    
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level[cmd_args.logging_level], format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    if cmd_args.ngpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.gpus)
        logger.info('Set CUDA_VISIBLE_DEVICES as %s' % cmd_args.gpus)

    if cmd_args.parallel_mode == 'ddp':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23456'
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.cuda.set_device(cmd_args.local_rank)

    main(cmd_args, params, expdir)


