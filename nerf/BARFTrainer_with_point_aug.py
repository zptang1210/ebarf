import os
import glob
import pickle
import torch
import tqdm
import tensorboardX
import cv2

# from .utils import Trainer, srgb_to_linear
from utils.event_utils import *
from .BARFTrainer import BARFTrainer
from .network_barf_with_point_aug import BARFNetwork_with_point_aug
from .network_barf import BARFNetwork
from torch_ema import ExponentialMovingAverage

class CheckLossOnPlateau:
    def __init__(self, max_stuck_times, patience=100, threshold=1e-4, threshold_mode='abs', cooldown=0, eps=1e-8, stuck_times=0):
        assert stuck_times <= max_stuck_times, 'stuck_times cannot be larger than max_stuck_times'
        self.stuck_times = stuck_times
        self.eps = eps
        self.dummy_model = torch.nn.Linear(1, 1, bias=False, device='cpu')
        self.dummy_optimizer = torch.optim.SGD(self.dummy_model.parameters(), lr=2**(max_stuck_times - stuck_times))
        self.dummy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.dummy_optimizer,
                                                                          mode='min',
                                                                          factor=1/2,
                                                                          patience=patience,
                                                                          threshold=threshold,
                                                                          threshold_mode=threshold_mode,
                                                                          cooldown=cooldown,
                                                                          min_lr=1.,
                                                                          eps=eps,
                                                                          verbose=False)
    
    def step(self, metrics):
        prev_dummy_lr = self.dummy_optimizer.param_groups[0]['lr']
        self.dummy_scheduler.step(metrics)
        curr_dummy_lr = self.dummy_optimizer.param_groups[0]['lr']

        if abs(prev_dummy_lr - curr_dummy_lr) > self.eps:
            self.stuck_times += 1
            return True
        else:
            return False
    
    def get_stuck_times_on_plateau(self):
        return self.stuck_times
    
    def state_dict(self):
        state = {'stuck_times': self.stuck_times,
                 'eps': self.eps,
                 'model': self.dummy_model.state_dict(),
                 'optim': self.dummy_optimizer.state_dict(),
                 'scheduler': self.dummy_scheduler.state_dict()}
        return state
    
    def load_state_dict(self, state):
        self.stuck_times = state['stuck_times']
        self.eps = state['eps']
        self.dummy_model.load_state_dict(state['model'])
        self.dummy_optimizer.load_state_dict(state['optim'])
        self.dummy_scheduler.load_state_dict(state['scheduler'])


class BARFTrainer_with_point_aug(BARFTrainer):
    def __init__(self,
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 max_pt_aug_times, # max time of point augmentation
                 optimizer, # optimizer
                 optimizer_pose, # optimizer for poses
                 lr_scheduler, # scheduler
                 lr_scheduler_pose, # scheduler for poses
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 ema_decay=None, # if use EMA, set the decay
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=True, # whether to call scheduler.step() after every train step 
                 scheduler_pose_update_every_step=False # whether to call scheduler_pose.step() after every train step 
                 ):
        self.max_pt_aug_times = max_pt_aug_times
        self.check_loss_on_plateau = CheckLossOnPlateau(max_pt_aug_times)        
        super().__init__(name,
                         opt,
                         model,
                         optimizer,
                         optimizer_pose,
                         lr_scheduler,
                         lr_scheduler_pose,
                         criterion=criterion,
                         ema_decay=ema_decay,
                         metrics=metrics,
                         local_rank=local_rank,
                         world_size=world_size,
                         device=device,
                         mute=mute,
                         fp16=fp16,
                         max_keep_ckpt=max_keep_ckpt,
                         best_mode=best_mode,
                         use_loss_as_metric=use_loss_as_metric,
                         report_metric_at_train=report_metric_at_train,
                         use_checkpoint=use_checkpoint,
                         use_tensorboardX=use_tensorboardX,
                         scheduler_update_every_step=scheduler_update_every_step,
                         scheduler_pose_update_every_step=scheduler_pose_update_every_step
                         )
        
    def __del__(self):
        super().__del__()

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.nerf.cuda_ray:
            raise NotImplementedError('cuda_ray not supported yet.')
            # self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
        
        self.error_map = train_loader._data.error_map
        assert self.error_map is None, 'error_map is not supported yet.'

        # * get a ref to evs_timespan_us
        self.evs_timespan_us = train_loader._data.evs_timespan_us

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            do_aug = self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)
            
            if do_aug:
                num_stuck_times = self.check_loss_on_plateau.get_stuck_times_on_plateau()

                # record the model and do evaluation before augmentation
                print(f'[AUG_INFO] Evaluate before augmentation {num_stuck_times}...')
                name = f'last_epoch_{self.epoch:08d}_before_aug_{num_stuck_times}'
                self.save_checkpoint(full=True, best=False, remove_old=False, name=name)
                self.evaluate_one_epoch(valid_loader, name=name)

                # point augmentation
                aug_model = BARFNetwork_with_point_aug(self.model)
                self.reset_model(aug_model)
                print(f'[AUG_INFO] Point Augmentation (#aug = {num_stuck_times}| #poses_hf after aug = {aug_model.tss_poses_hf_ns.shape[0]})')
                if num_stuck_times >= self.max_pt_aug_times: print(f'[AUG_INFO] Reached the max_pt_aug_times ({self.max_pt_aug_times})')
                self.check_loss_on_plateau = CheckLossOnPlateau(self.max_pt_aug_times, stuck_times=num_stuck_times, patience=self.opt.aug_patience)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def reset_model(self, new_model):
        new_model.to(self.device)
        if self.world_size > 1:
            new_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(new_model)
            new_model = torch.nn.parallel.DistributedDataParallel(new_model, device_ids=[self.local_rank])

        # todo try different strategies.
        # reset the lr for both nerf and se3_refine
        optimizer, lr_scheduler, optimizer_pose, scheduler_pose, scheduler_update_every_step, scheduler_pose_update_every_step = __class__.get_optimizer_and_scheduler(self.opt.lr, self.opt.lr_pose, self.opt.iters-self.global_step, self.opt)
        assert self.scheduler_update_every_step == scheduler_update_every_step
        assert self.scheduler_pose_update_every_step == scheduler_pose_update_every_step

        self.model = new_model
        self.optimizer_lambdafunc = optimizer
        self.lr_scheduler_lambdafunc = lr_scheduler
        self.optimizer_pose_lambdafunc = optimizer_pose
        self.lr_scheduler_pose_lambdafunc = scheduler_pose
        self.init_optimizer()

        if self.ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
        else:
            self.ema = None

    @classmethod
    def get_optimizer_and_scheduler(cls, lr, lr_pose, total_iters, opt):
        optimizer, scheduler = cls._get_optimizer_and_scheduler_for_nerf(lr, total_iters)
        optimizer_pose, scheduler_pose = cls._get_optimizer_and_scheduler_for_pose(lr_pose, opt.aug_patience)
        scheduler_update_every_step, scheduler_pose_update_every_step = True, False
        return optimizer, scheduler, optimizer_pose, scheduler_pose, scheduler_update_every_step, scheduler_pose_update_every_step

    @classmethod
    def _get_optimizer_and_reduce_lr_on_plateau_scheduler_for_pose(cls, lr_pose, aug_patience, patience=10, min_lr_scale=0.1):
        optimizer_pose = lambda model: torch.optim.Adam(model.se3_refine.parameters(), lr=lr_pose)

        min_lr = lr_pose * min_lr_scale
        assert math.ceil(aug_patience / patience) - 1 > 0, 'aug_patience should be at least two times larger than patience'
        factor = math.pow(min_lr / lr_pose, 1 / (math.ceil(aug_patience / patience) - 1))
        scheduler_pose = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
        return optimizer_pose, scheduler_pose
    
    _get_optimizer_and_scheduler_for_pose = _get_optimizer_and_reduce_lr_on_plateau_scheduler_for_pose

    def train_one_epoch(self, loader):
        average_loss = super().train_one_epoch(loader)
        on_plateau = self.check_loss_on_plateau.step(average_loss)
        return on_plateau

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'model.out_dim_color': self.model.out_dim_color,
            'model.raw_poses_hf': self.model.raw_poses_hf,
            'model.raw_tss_poses_hf_ns': self.model.raw_tss_poses_hf_ns,
            'model.poses_hf.shape': self.model.poses_hf.shape,
            'model.tss_poses_hf_ns.shape': self.model.tss_poses_hf_ns.shape
        }

        if self.model.nerf.cuda_ray:
            state['mean_count'] = self.model.nerf.mean_count
            state['mean_density'] = self.model.nerf.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['optimizer_pose'] = self.optimizer_pose.state_dict()
            state['lr_scheduler_pose'] = self.lr_scheduler_pose.state_dict()
            state['check_loss_on_plateau'] = self.check_loss_on_plateau.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                if self.render_mode:
                    sys.exit()
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        # create a dummy model so that the shape of it is the same as the shape of the one stored in the checkpoint.
        tss_poses_hf_ns_shape = checkpoint_dict['model.tss_poses_hf_ns.shape']
        poses_hf_shape = checkpoint_dict['model.poses_hf.shape']
        dummy_poses_hf_dict = {'tss_poses_hf_ns': torch.zeros(tss_poses_hf_ns_shape),
                               'poses_hf': torch.zeros(poses_hf_shape),
                               'raw_tss_poses_hf_ns': checkpoint_dict['model.raw_tss_poses_hf_ns'],
                               'raw_poses_hf': checkpoint_dict['model.raw_poses_hf']}
        dummy_intrinsics_evs = torch.zeros_like(self.model.intrinsics_evs)
        dummy_model = BARFNetwork(dummy_poses_hf_dict, dummy_intrinsics_evs)
        dummy_model.nerf = self.model.nerf

        self.reset_model(dummy_model)
        
        if 'model' not in checkpoint_dict:
            raise KeyError("checkpoint_dict does not have key 'model'")
            # self.model.load_state_dict(checkpoint_dict)
            # self.log("[INFO] loaded model.")
            # return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.nerf.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.nerf.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.nerf.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.model.out_dim_color = checkpoint_dict['model.out_dim_color']
        self.model.raw_poses_hf = checkpoint_dict['model.raw_poses_hf']
        self.model.raw_tss_poses_hf_ns = checkpoint_dict['model.raw_tss_poses_hf_ns']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log(f"[INFO] loaded optimizer. (lr={self.optimizer.param_groups[0]['lr']})")
            except Exception as error:
                self.log("[WARN] Failed to load optimizer.", error)
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except Exception as error:
                self.log("[WARN] Failed to load scheduler.", error)

        if self.optimizer_pose and  'optimizer_pose' in checkpoint_dict:
            try:
                self.optimizer_pose.load_state_dict(checkpoint_dict['optimizer_pose'])
                self.log("[INFO] loaded optimizer_pose. (lr_pose={self.optimizer_pose.param_groups[0]['lr']})")
            except Exception as error:
                self.log("[WARN] Failed to load optimizer_pose.", error)
        
        if self.lr_scheduler_pose and 'lr_scheduler_pose' in checkpoint_dict:
            try:
                self.lr_scheduler_pose.load_state_dict(checkpoint_dict['lr_scheduler_pose'])
                self.log("[INFO] loaded scheduler_poses.")
            except Exception as error:
                self.log("[WARN] Failed to load scheduler_poses.", error)
        
        if self.check_loss_on_plateau and 'check_loss_on_plateau' in checkpoint_dict:
            try:
                self.check_loss_on_plateau.load_state_dict(checkpoint_dict['check_loss_on_plateau'])
                self.log(f"[INFO] loaded check_loss_on_plateau. (stuck_times={self.check_loss_on_plateau.get_stuck_times_on_plateau()})")
            except Exception as error:
                self.log("[WARN] Failed to load check_loss_on_plateau.", error)
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except Exception as error:
                self.log("[WARN] Failed to load scaler.", error)