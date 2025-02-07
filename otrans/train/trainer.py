import os
import torch
import math
import time
import logging
import torch.distributed as dist
from otrans.train.utils import MeanLoss, Visulizer, AverageMeter, Summary, map_to_cuda, AuxiliaryLossAverageMeter
logger = logging.getLogger(__name__)
from  tensorboardX import SummaryWriter
import time
try:
    from frob import frobdecay
except ImportError:
    print("Failed to import Frobenius decay")

class Trainer(object):
    def __init__(self, args,params, model, optimizer, scheduler, skiplist=[],is_visual=False,
                 expdir='./', ngpu=1, parallel_mode='dp', local_rank=0, is_debug=False,
                 keep_last_n_chkpt=30, from_epoch=0):

        self.params = params
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.expdir = expdir
        self.is_visual = is_visual
        self.is_debug = is_debug
        self.args=args
        self.ngpu = ngpu
        self.parallel_mode = parallel_mode
        self.local_rank = local_rank
        self.from_epoch = from_epoch
        self.skiplist=skiplist
        self.total_epochs = params['train']['epochs']
        self.accum_steps = params['train']['accum_steps']
        self.grad_noise = params['train']['grad_noise']

        self.grad_clip = params['train']['clip_grad']
        self.global_training_step = 1
        self.log_interval = 10
        self.mean_loss = MeanLoss()
        self.training_infos = ['**Training INFO**']

        self.keep_last_n_chkpt = keep_last_n_chkpt

        if self.is_visual and local_rank == 0:
            self.visulizer = Visulizer(log_dir=os.path.join(expdir, 'visual'))

        if self.params['train']['load_model']:
            self.load_model(self.params['train']['load_model'])
            logger.info('Load the checkpoint from %s' % self.params['train']['load_model'])
            self.training_infos.append('Load the checkpoint from %s' % self.params['train']['load_model'])

        if self.ngpu > 1:
            if self.parallel_mode == 'ddp':
                import torch.distributed as dist
                dist.init_process_group(backend="nccl", world_size=self.ngpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
                logger.info('[DDP] Use %d gpus for training!' % self.ngpu)

            elif self.parallel_mode == 'dp':
                self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(self.ngpu)])
                logger.info('[DP] Use %d gpus for training!' % self.ngpu)

            else:
                logging.warning('Please chose one of dp, ddp and hvd for parallel computing!')

        elif self.ngpu == 1:
            logger.info('Use only 1 gpu for training!')
        else:
            logger.info('Train the model in CPU!')

        if self.grad_noise > 0.0:
            logger.info('Apply Grad Noise mean = 0.0, std = %f' % self.grad_noise)
            self.training_infos.append(['Apply Grad Noise mean = 0.0, std = %f' % self.grad_noise])

    def train(self, train_loader, dev_loader=None):

        self.training_infos.append('Begining to Train...')
        epochs = self.params['train']['epochs']
        TrainLossNote = Summary()
        DevLossNote = Summary()
        if self.local_rank == 0:
            self.writer=SummaryWriter('./logs/'+time.asctime())

        for epoch in range(self.from_epoch, self.total_epochs):

            train_loader.set_epoch(epoch)

            if self.ngpu > 1:
                self.model.module.set_epoch(epoch)
            else:
                self.model.set_epoch(epoch)
            # self.optimizer.epoch()
 
            train_loss = self.train_one_epoch(epoch, train_loader.loader)
            TrainLossNote.update(epoch, train_loss)
            self.scheduler.epoch()

            if self.local_rank == 0:
                logger.info('-*Train-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, train_loss))
                self.training_infos.append('-*Train-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, train_loss))

                self.save_model(epoch)
                self.save_optimizer_state_dict()
                # auto clear the redundant checkpoints to save memory
                self.clear_checkpoint(epoch)

            if self.is_visual and self.local_rank == 0:
                self.visulizer.add_scalar('train_epoch_loss', train_loss, epoch)

            if dev_loader is not None:
                dev_loss = self.eval(dev_loader.loader)
                DevLossNote.update(epoch, dev_loss)
                if self.local_rank == 0:
                    logger.info('-*Eval-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, dev_loss))
                    self.training_infos.append('-*Eval-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, dev_loss))
                
                if dev_loss < DevLossNote.best()[1] and self.local_rank == 0:
                    self.save_model('model.best.pt')
                    logger.info('Update the best checkpoint!')

        self.optimizer.zero_grad() # clean the residual grad

        if self.local_rank == 0:
            logger.info('Training Summary:')
            BEST_T_EPOCH, BEST_T_LOSS = TrainLossNote.best()
            logger.info('At the %d-st epoch of training, the model performs best (Loss:%.5f)!' % (BEST_T_EPOCH, BEST_T_LOSS))
            self.training_infos.append('At the %d-st epoch of training, the model performs best (Loss:%.5f)!' % (BEST_T_EPOCH, BEST_T_LOSS))
            
            if dev_loader is not None:
                BEST_E_EPOCH, BEST_E_LOSS = DevLossNote.best()
                logger.info('At the %d-st epoch of validation, the model performs best (Loss:%.5f)!' % (BEST_E_EPOCH, BEST_E_LOSS))
                self.training_infos.append('At the %d-st epoch of validation, the model performs best (Loss:%.5f)!' % (BEST_E_EPOCH, BEST_E_LOSS))
            
            if self.is_visual:
                self.visulizer.close()

    def train_one_epoch(self, epoch, train_loader):

        self.model.train()
        batch_steps = len(train_loader)

        step_loss = AverageMeter()
        auxiliary_loss = AuxiliaryLossAverageMeter()
        span = 0
        grad_denom = batch_steps
        for step, (_, inputs, targets) in enumerate(train_loader):

            if self.ngpu > 0:
                inputs = map_to_cuda(inputs)
                targets = map_to_cuda(targets)

            start = time.time()
            
            # loss: tensor
            # axu_loss: dict {loss1: value1, loss2: value2}
            # self.model.forward_hook(self.scheduler.global_step, self.scheduler.global_epoch)
            loss, aux_loss = self.model(inputs, targets)
            loss = torch.mean(loss) / self.accum_steps
            loss.backward()
            end = time.time()
            span += (end - start)
            if self.get_rank() == 0:
                step_loss.update(loss.item() * self.accum_steps, inputs['inputs'].size(0))
                auxiliary_loss.update(aux_loss, self.accum_steps, inputs['inputs'].size(0))

            if self.global_training_step % self.accum_steps == 0:
                if self.local_rank == 0:
                    self.mean_loss.update(step_loss.avg)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                if self.grad_noise > 0.0:
                    for p in self.model.parameters():
                        if p.requires_grad:
                            noise = torch.normal(0, self.grad_noise, p.grad.shape, device=loss.device)
                            p.grad += noise / self.accum_steps

                if math.isnan(grad_norm):
                    logging.warning('Grad norm is NAN. DO NOT UPDATE MODEL!')
                else:
                    self.scheduler.step()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.is_visual and self.local_rank == 0:
                    self.visulizer.add_scalar('train_loss', loss.item(), self.scheduler.global_step)
                    self.visulizer.add_scalar('lr', self.scheduler.lr, self.scheduler.global_step)

                if self.scheduler.global_step % self.log_interval == 0 and self.local_rank == 0:
                    process = (step + 1) / batch_steps * 100
                    print_info = "-Training-Epoch-%d(%.5f%%), Global Step:%d, lr:%.8f, Loss:%.5f, AvgLoss: %.5f, Run Time:%.3f" \
                        % (epoch, process, self.scheduler.global_step, self.scheduler.lr, step_loss.avg, self.mean_loss.mean(), span)
                    print_info += auxiliary_loss.avg_infos
                    logger.info(print_info)
                    self.writer.add_scalar('loss/epoc', step_loss.avg, epoch*batch_steps+step)
                    self.writer.flush()
                    span = 0
                step_loss.reset()
                auxiliary_loss.reset()
            self.global_training_step += 1

            if self.is_debug and step > 30:
                break

        return self.mean_loss.mean()

    def eval(self, dev_loader):
        self.model.eval()
        eval_loss = 0
        for step, (_, batch,targets) in enumerate(dev_loader):
            if self.ngpu > 0:
                batch = map_to_cuda(batch)
                targets = map_to_cuda(targets)
            loss,_= self.model(batch, targets)
            loss = torch.mean(loss)
            eval_loss += loss.item()

        return eval_loss / (step+1)

    def save_model(self, epoch=None, save_name=None):
        if save_name is None:
            save_name = 'model.epoch.%d.%f.pt' % (epoch,self.args.rank_scale)

        if self.ngpu > 1:
            self.model.module.save_checkpoint(self.params, os.path.join(self.expdir, save_name))
        else:
            self.model.save_checkpoint(self.params, os.path.join(self.expdir, save_name))
        logger.info('Save the model checkpoint!')

    def save_optimizer_state_dict(self, save_name=None):
        if save_name is None:
            save_name = 'latest_optimizer.pt'

        checkpoint = {
            'global_step': self.scheduler.global_step,
            'optim': self.optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(self.expdir, save_name))
        logger.info('Save the optimizer checkpoint!')

    def clear_checkpoint(self, epoch):
        if epoch + 1 > self.keep_last_n_chkpt:
            save_name = 'model.epoch.%d.%f.pt' % (epoch - self.keep_last_n_chkpt,self.args.rank_scale)
            if os.path.isfile(os.path.join(self.expdir, save_name)):
                os.remove(os.path.join(self.expdir, save_name))
                logger.info('Delete the checkpoint %s to save memory!' % os.path.join(self.expdir, save_name))
        else:
            logger.info('There are no any checkpoints to be cleaned!')

    def load_model(self, checkpoint):
        chkpt = torch.load(checkpoint)
        if 'model' in chkpt:
            self.model.load_state_dict(chkpt['model'])
        elif 'encoder' in chkpt and 'decoder' in chkpt:
            self.model.encoder.load_state_dict(chkpt['encoder'])
            self.model.decoder.load_state_dict(chkpt['decoder'])

        if 'ctc' in chkpt:
            self.model.assistor.load_state_dict(chkpt['ctc'])

    def get_rank(self):

        if self.parallel_mode == 'ddp':
            return dist.get_rank()
        else:
            return 0
        
