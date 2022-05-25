#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn

import torch.distributed as dist
#import thop
# import torch.nn.parallel.DistributedDataParallel as DDP


def init_seed(_):
    torch.cuda.manual_seed_all(_)
    torch.manual_seed(_)
    np.random.seed(_)
    random.seed(_)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Multi-Scale Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='./test/')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save_score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start_epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='checkpoint path')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
                    
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        if arg.local_rank == 0:
            self.save_arg()
            if arg.phase == 'train':
                if not arg.train_feeder_args['debug']:
                    if os.path.isdir(arg.model_saved_name):
                        print('log_dir: ', arg.model_saved_name, 'already exist')
                        answer = input('delete it? y/n:')
                        if answer == 'y':
                            shutil.rmtree(arg.model_saved_name)
                            print('Dir removed: ', arg.model_saved_name)
                            input('Refresh the website of tensorboard by pressing any keys')
                        else:
                            print('Dir not removed: ', arg.model_saved_name)
                    self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                    self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
                else:
                    self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = [0, 0] # [epoch, acc]

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.train_data = Feeder(**self.arg.train_feeder_args)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=self.arg.batch_size,
                shuffle=False,
                pin_memory=True,
                sampler=self.train_sampler,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.test_data = Feeder(**self.arg.test_feeder_args)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_data)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            pin_memory=True,
            sampler=self.test_sampler,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
#        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
#        self.output_device = output_device

        output_device = torch.device(f'cuda:{arg.local_rank}')
        self.output_device = output_device
        
        Model = import_class(self.arg.model)
        if arg.local_rank == 0:
            shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        #print(Model)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Model(**self.arg.model_args)).cuda(output_device)
        # self.model = Model(**self.arg.model_args).cuda(output_device)
        self.print_log('# generator parameters: {} M.'.format(
            sum(param.numel() for param in self.model.parameters())/10**6))
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.arg.local_rank],
                                                               find_unused_parameters=True)
          

#        if type(self.arg.device) is list:
#            if len(self.arg.device) > 1:
#                self.model = nn.DataParallel(
#                    self.model,
#                    device_ids=self.arg.device,
#                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

            if self.arg.start_epoch != 0:
                checkpoint = torch.load(self.arg.model_path)
                self.optimizer.load_state_dict(checkpoint['optimizer'])    
                self.print_log('Sucessfully Load Optimizer from: {}.'.format(self.arg.model_path))          
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_arg(self):
        # save arg
        if arg.local_rank == 0:
            arg_dict = vars(self.arg)
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
                yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        if arg.local_rank == 0:
            localtime = time.asctime(time.localtime(time.time()))
            self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if arg.local_rank == 0:
            if print_time:
                localtime = time.asctime(time.localtime(time.time()))
                str = "[ " + localtime + ' ] ' + str
            print(str)
            if self.arg.print_log:
                with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                    print(str, file=f)

    def record_time(self):
        if arg.local_rank == 0:
            self.cur_time = time.time()
            return self.cur_time

    def split_time(self):
        if arg.local_rank == 0:
            split_time = time.time() - self.cur_time
            self.record_time()
            return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        self.print_log('Learning rate: {}'.format(self.optimizer.param_groups[0]['lr']))
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        if arg.local_rank == 0:
            self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = data.float().cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            if arg.local_rank == 0:
                timer['dataloader'] += self.split_time()

            # forward
            features, output = self.model(data)
            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())
            if arg.local_rank == 0:
                timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            if arg.local_rank == 0:
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.arg.local_rank == 0:
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            if arg.local_rank == 0:
                timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        if arg.local_rank == 0:
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        if arg.local_rank == 0:
            self.print_log(
                '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                    **proportion))
            if save_model:
                if arg.local_rank == 0:
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1],
                                            v.cpu()] for k, v in state_dict.items()])
        
                    torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')         

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        with torch.no_grad():
            self.model.eval()
            self.print_log('Eval epoch: {}'.format(epoch + 1))
            for ln in loader_name:
                loss_value = []
                score_frag = []
                right_num_total = 0
                total_num = 0
                loss_total = 0
                step = 0
                process = tqdm(self.data_loader[ln])
                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    # if isinstance(output, tuple):
                    #     output, l1 = output
                    #     l1 = l1.mean()
                    # else:
                    #     l1 = 0
                    # loss = self.loss(output, label)
                    # score_frag.append(output.data.cpu().numpy())
                    # loss_value.append(loss.item())
    
                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    right_num = torch.sum(predict_label == label.data).item()
                    batch_num = label.data.size(0)
                    right_num_total += right_num
                    total_num += batch_num
    
                    # if wrong_file is not None or result_file is not None:
                    #     predict = list(predict_label.cpu().numpy())
                    #     true = list(label.data.cpu().numpy())
                    #     for i, x in enumerate(predict):
                    #         if result_file is not None:
                    #             f_r.write(str(x) + ',' + str(true[i]) + '\n')
                    #         if x != true[i] and wrong_file is not None:
                    #             f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
                # score = np.concatenate(score_frag)
                # loss = np.mean(loss_value)
                # accuracy = self.data_loader[ln].dataset.top_k(score, 1)

                right_num_total = torch.Tensor([right_num_total]).cuda(arg.local_rank)
                dist.all_reduce(right_num_total)
                right_num_total = int(right_num_total)

                total_num = torch.Tensor([total_num]).cuda(arg.local_rank)
                dist.all_reduce(total_num)
                total_num = int(total_num)

                accuracy = int(right_num_total) / total_num

                if accuracy > self.best_acc[1]:
                    self.best_acc[0] = epoch + 1
                    self.best_acc[1] = accuracy
                # self.lr_scheduler.step(loss)
                if arg.local_rank == 0:
                    self.print_log('Epoch {} Curr Acc: ({}/{}){:.2f}%'.format(epoch + 1, right_num_total, total_num,
                                                                                         accuracy*100))
                    self.print_log('Epoch {} Best Acc {:.2f}%'.format(self.best_acc[0], self.best_acc[1]*100))
                    if self.arg.phase == 'train':
                        # self.val_writer.add_scalar('loss', loss, self.global_step)
                        # self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                        self.val_writer.add_scalar('acc', accuracy, self.global_step)
    
                # score_dict = dict(
                #     zip(self.data_loader[ln].dataset.sample_name, score))
                # self.print_log('\tMean {} loss of {} batches: {}.'.format(
                #     ln, len(self.data_loader[ln]), np.mean(loss_value)))
                # for k in self.arg.show_topk:
                #     self.print_log('\tTop{}: {:.2f}%'.format(
                #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
                #
                # if save_score:
                #     with open('{}/epoch{}_{}_score.pkl'.format(
                #             self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                #         pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
#                if self.lr < 1e-3:
#                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0 and (epoch + 1) > 50) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)
                if (epoch + 1) > self.arg.step[0]:
                    self.eval(
                        epoch,
                        save_score=self.arg.save_score,
                        loader_name=['test'])

            self.print_log('epoch: {}, best accuracy: {}'.format(self.best_acc[0], self.best_acc[1]))
            self.print_log('Experiment: {}'.format(self.arg.work_dir))

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    torch.cuda.set_device(arg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
