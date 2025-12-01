import os
import sys
import numpy as np
import datetime
from tqdm import tqdm
import torch
from copy import deepcopy
import json
from torch.utils.data import DataLoader
from monai.inferers import SlidingWindowInferer

from configs_expert_pretraining import TrainingConfig


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n' + '==========' * 8 + '%s' % nowtime)
    print(str(info) + '\n')


class StepRunner:
    def __init__(self, net, loss_fn,
                 stage='train', metrics_dict=None,
                 optimizer=None,
                 device='cuda:0'
                 ):
        self.net = net
        self.loss_fn, self.metrics_dict, self.stage = loss_fn, metrics_dict, stage
        self.optimizer = optimizer
        self.device = device

    def step(self, batch):
        # compute loss
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        preds = self.net(features)
        if TrainingConfig.APPLY_SAMPLE_WEIGHTS:
            weights = batch['weight'].to(self.device)
            loss = self.loss_fn(preds, labels, weights)
        else:
            loss = self.loss_fn(preds, labels)

        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), len(labels)

    def step_infer(self, batch):
        # inferer = SlidingWindowInferer(roi_size=[128] * 3, progress=False)
        # preds = inferer(features, self.net)
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        preds = self.net(features)
        if TrainingConfig.APPLY_SAMPLE_WEIGHTS:
            weights = batch['weight'].to(self.device)
            loss = self.loss_fn(preds, labels, weights)
        else:
            loss = self.loss_fn(preds, labels)

        step_metrics = {self.stage + '_' + name: metric_fn(preds, labels).item()
                        for name, metric_fn in self.metrics_dict.items()}

        return loss.item(), step_metrics, len(labels)

    def train_step(self, batch):
        self.net.train()

        return self.step(batch)

    @torch.no_grad()
    def eval_step(self, batch):
        self.net.eval()

        return self.step_infer(batch)

    def __call__(self, batch):
        if self.stage == 'train':
            return self.train_step(batch)
        else:
            return self.eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner: StepRunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def _train_handler(self):
        cur_loss = 0
        cur_num_sample = 0
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            step_loss, batch_size = self.steprunner(batch)
            cur_loss = (cur_num_sample * cur_loss + batch_size * step_loss) / (cur_num_sample + batch_size)
            cur_num_sample += batch_size

            cur_log = dict({self.stage + '_loss': cur_loss})
            loop.set_postfix(**cur_log)

        return cur_log

    def _eval_handler(self):
        cur_loss = 0
        cur_num_sample = 0
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            step_loss, step_metrics, batch_size = self.steprunner(batch)

            # compute current loss & metrics
            cur_loss = (cur_num_sample * cur_loss + batch_size * step_loss) / (cur_num_sample + batch_size)
            if step == 0:
                cur_metrics = step_metrics.copy()
            else:
                for metric_name, metric_value in step_metrics.items():
                    cur_metrics[metric_name] = (cur_num_sample * cur_metrics[
                        metric_name] + batch_size * metric_value) / (cur_num_sample + batch_size)
            cur_num_sample += batch_size

            cur_log = dict({self.stage + '_loss': cur_loss}, **cur_metrics)
            loop.set_postfix(**cur_log)

        return cur_log

    def __call__(self, dataloader: DataLoader):
        self.dataloader = dataloader

        if self.stage == 'train':
            cur_log = self._train_handler()
        else:
            cur_log = self._eval_handler()

        return cur_log


def train_model(net, optimizer, loss_fn, metrics_dict, train_data, val_data, val_freq=1,
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
            loss_fn=loss_fn, metrics_dict=None,
            optimizer=optimizer, device=device
        )
        train_epoch_runner = EpochRunner(train_step_runner)
        train_log = train_epoch_runner(train_data)

        for name, metric in train_log.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if epoch % val_freq == 0 and val_data:
            val_step_runner = StepRunner(
                net=net, stage='val', loss_fn=loss_fn,
                metrics_dict=deepcopy(metrics_dict), device=device
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
