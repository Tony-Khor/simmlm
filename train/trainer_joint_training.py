import os
import sys
import numpy as np
import datetime
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader
from monai.metrics import DiceHelper

from configs_joint_training import ModelConfig
from loss.blended_loss import BlendedLoss


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n' + '==========' * 8 + '%s' % nowtime)
    print(str(info) + '\n')


class StepRunner:
    def __init__(self, net,
                 stage='train',
                 optimizer=None,
                 device='cuda:0'
                 ):
        self.net = net
        self.stage = stage
        self.optimizer = optimizer
        self.device = device
        self.train_loss_fn = BlendedLoss(**ModelConfig.TRAIN_LOSS_ARGS)
        self.val_loss_fn = BlendedLoss(**ModelConfig.VAL_LOSS_ARGS)
        self.dice_fn = DiceHelper(
            include_background=True, sigmoid=True, softmax=False,
            activate=ModelConfig.TRAIN_LOSS_ARGS['need_sigmoid'], get_not_nans=False, reduction='none'
        )

    def _train_step(self, batch, epoch):
        self.net = self.net.train()
        # compute loss
        features = torch.cat((batch['img'].to(self.device), batch['img_'].to(self.device)))
        labels = torch.cat((batch['label'].to(self.device), batch['label_'].to(self.device)))

        preds = self.net(features)

        loss_dict = self.train_loss_fn(preds, labels)

        # backward
        loss_dict['task_loss'].backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        for key in loss_dict:
            loss_dict[key] = loss_dict[key].item()
        return loss_dict, len(labels)

    @torch.no_grad()
    def _eval_step(self, batch):
        self.net = self.net.eval()
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        preds = self.net(features)

        dice = self.dice_fn(preds, labels)
        mean_dice = torch.nanmean(dice)

        loss_dict = self.val_loss_fn(preds, labels)

        metric_dict = {
            'val_loss': loss_dict['task_loss'].item(),
            'val_dice': mean_dice.item()
        }

        return metric_dict, len(labels)

    def __call__(self, batch, epoch=0):
        if self.stage == 'train':
            return self._train_step(batch, epoch)
        else:
            return self._eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner: StepRunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def _train_handler(self, epoch):

        cur_num_sample = 0
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            step_loss_dict, batch_size = self.steprunner(batch, epoch)

            if step == 0:
                cur_log = step_loss_dict.copy()
            else:
                for loss_name, loss_value in step_loss_dict.items():
                    cur_log[loss_name] = (cur_num_sample * cur_log[
                        loss_name] + batch_size * loss_value) / (cur_num_sample + batch_size)

            cur_num_sample += batch_size

            loop.set_postfix(**cur_log)

        return cur_log

    def _eval_handler(self):
        cur_num_sample = 0
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            step_metric_dict, batch_size = self.steprunner(batch)

            # compute current metrics
            if step == 0:
                cur_log = step_metric_dict.copy()
            else:
                for metric_name, metric_value in step_metric_dict.items():
                    cur_log[metric_name] = (cur_num_sample * cur_log[
                        metric_name] + batch_size * metric_value) / (cur_num_sample + batch_size)

            cur_num_sample += batch_size

            loop.set_postfix(**cur_log)

        return cur_log

    def __call__(self, dataloader: DataLoader, epoch=0):
        self.dataloader = dataloader

        if self.stage == 'train':
            cur_log = self._train_handler(epoch)
        else:
            cur_log = self._eval_handler()

        return cur_log


def train_model(net, optimizer, train_data, val_data, val_freq=1,
                num_epoch=100, results_dir='results/0/', device='cuda:0',
                apply_early_stopping=False, patience=5, monitor='val_loss', eval_mode='min'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    history = {}

    for epoch in range(1, num_epoch + 1):
        printlog('Epoch {0} / {1}'.format(epoch, num_epoch))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(
            net=net, stage='train',
            optimizer=optimizer, device=device
        )
        train_epoch_runner = EpochRunner(train_step_runner)
        train_log = train_epoch_runner(train_data, epoch)

        for name, metric in train_log.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if epoch % val_freq == 0 and val_data:
            val_step_runner = StepRunner(
                net=net, stage='val',
                device=device
            )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_log = val_epoch_runner(val_data)
            val_log['val_epoch'] = epoch
            for name, metric in val_log.items():
                history[name] = history.get(name, []) + [metric]

            # 3，eval best model & early-stopping -------------------------------------------------
            arr_scores = history[monitor]
            best_score_idx = np.argmax(arr_scores) if eval_mode == 'max' else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores) - 1:
                torch.save(net.state_dict(), os.path.join(results_dir, 'ckpt_bst.pt'))
                print('<<<<<< reach best {0} : {1} >>>>>>'.format(
                    monitor, arr_scores[best_score_idx]), file=sys.stderr
                )
            if apply_early_stopping and len(arr_scores) - best_score_idx > patience:
                print('<<<<<< {} without improvement in {} turns of validations, early stopping >>>>>>'.format(
                    monitor, patience), file=sys.stderr)
                break

            with open(os.path.join(results_dir, 'hist.json'), 'w') as outfile:
                json.dump(history, outfile)

    torch.save(net.state_dict(), os.path.join(results_dir, 'ckpt_final.pt'))

    print(history)

    return history
