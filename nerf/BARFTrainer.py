import os
import glob
import pickle
import torch
from torch import optim
import tqdm
import tensorboardX
import cv2

from .utils import Trainer, srgb_to_linear
from utils.event_utils import *

class BARFTrainer(Trainer):
    def __init__(self,
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
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
                 scheduler_pose_update_every_step=True # whether to call scheduler_pose.step() after every train step 
                 ):
        
        self.optimizer_pose_lambdafunc = optimizer_pose
        self.lr_scheduler_pose_lambdafunc = lr_scheduler_pose
        self.scheduler_pose_update_every_step = scheduler_pose_update_every_step
        super().__init__(name,
                         opt,
                         model,
                         criterion=criterion,
                         optimizer=optimizer,
                         ema_decay=ema_decay,
                         lr_scheduler=lr_scheduler,
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
                         scheduler_update_every_step=scheduler_update_every_step
                         )
        
    def init_optimizer(self):
        super().init_optimizer()
        self.optimizer_pose = self.optimizer_pose_lambdafunc(self.model)
        self.lr_scheduler_pose = self.lr_scheduler_pose_lambdafunc(self.optimizer_pose)        
    
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
        self.poses_hf_dict_final = train_loader._data.poses_hf_dict_final

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)


            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} lr_pose={self.optimizer_pose.param_groups[0]['lr']:.6f}...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        for data in loader:
            # update grid every 16 steps
            if self.model.nerf.cuda_ray and self.global_step % 16 == 0:
                raise NotImplementedError('cuda_ray not supported yet.')
                # with torch.cuda.amp.autocast(enabled=self.fp16):
                #     self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            self.optimizer_pose.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.use_events:
                    preds, truths, loss, est_C_thres, losses_indiv = self.train_step_events(data)
                else:
                    raise NotImplementedError('only support use_events=True.')
                    # preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.step(self.optimizer_pose)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            if self.scheduler_pose_update_every_step:
                self.lr_scheduler_pose.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/epoch", self.epoch, self.global_step)
                    if self.use_events:
                        self.writer.add_scalar("train/loss_evs", losses_indiv["loss_evs"].item(), self.global_step)
                        if self.negative_event_sampling and self.epoch > self.epoch_start_noEvLoss:
                            raise NotImplementedError('negative event sampling not supported yet.')
                            # self.writer.add_scalar("train/loss_no_evs", losses_indiv["loss_no_evs"].item(), self.global_step)
                        if not self.event_only:
                            raise NotImplementedError('only support event_only=True')
                            # self.writer.add_scalar("train/loss_frames", losses_indiv["loss_frames"].item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("train/lr_pose", self.optimizer_pose.param_groups[0]['lr'], self.global_step)
                    if self.log_implicit_C_thres and self.use_events:
                        self.writer.add_scalar("train/est_C_med_on", est_C_thres["median_on"], self.global_step)
                        self.writer.add_scalar("train/est_C_med_off", est_C_thres["median_off"], self.global_step)
                        self.writer.add_scalar("train/est_C_med_on_sign", est_C_thres["median_on_sign"], self.global_step)
                        self.writer.add_scalar("train/est_C_med_off_sign", est_C_thres["median_off_sign"], self.global_step)

                desc = f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                if self.scheduler_update_every_step:
                    desc += f", lr={self.optimizer.param_groups[0]['lr']:.6f}"
                if self.scheduler_pose_update_every_step:
                    desc += f", lr_pose={self.optimizer_pose.param_groups[0]['lr']:.6f}"
                pbar.set_description(desc)
                # self.log(desc)
                pbar.update(loader.batch_size)

                if self.epoch <= self.eval_interval:
                    save_path_gt = os.path.join(self.workspace, "validation", "gt_trainViews", f'{self.local_step-1:04d}_gt.png')
                    if not os.path.isdir(os.path.dirname(save_path_gt)):
                        os.makedirs(os.path.dirname(save_path_gt), exist_ok=True)
                    cv2.imwrite(save_path_gt, cv2.cvtColor((loader._data.images[self.local_step-1].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        if not self.scheduler_pose_update_every_step:
            if isinstance(self.lr_scheduler_pose, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler_pose.step(average_loss)
            else:
                self.lr_scheduler_pose.step()

        self.log(f"==> Finished Epoch {self.epoch}. Average loss = {average_loss}")
        return average_loss

    def train_step_events(self, data):
        loss_evs, loss_no_evs, loss_frames = -1,-1,-1  # init for logging

        outputs1, outputs2 = self.model(data)

        # Convert I => log(I)
        if self.use_luma:
            pred_luma1 = rgb_to_luma(outputs1["image"], esim=True) # (B, Nevs, 1)
            pred_luma2 = rgb_to_luma(outputs2["image"], esim=True) # (B, Nevs, 1)
            if self.linlog:
                pred_linlog1 = lin_log(pred_luma1*255, linlog_thres=20) # (B, Nevs, 1)
                pred_linlog2 = lin_log(pred_luma2*255, linlog_thres=20) # (B, Nevs, 1)
            else:
                pred_linlog1 = torch.log(torch.maximum(pred_luma1*255, self.log_thres))
                pred_linlog2 = torch.log(torch.maximum(pred_luma1*255, self.log_thres))
        else:
            if self.linlog:
                pred_linlog1 = lin_log(outputs1["image"]*255, linlog_thres=20) # (B, Nevs, 3)
                pred_linlog2 = lin_log(outputs2["image"]*255, linlog_thres=20) # (B, Nevs, 3)
            else:
                pred_linlog1 = torch.log(torch.maximum(outputs1["image"]*255, self.log_thres))
                pred_linlog2 = torch.log(torch.maximum(outputs2["image"]*255, self.log_thres))

        # Compute L2-L1 and pols
        w_evLoss = 1
        delta_linlog = (pred_linlog2 - pred_linlog1) # (B, Nevs, 1or3)
        gt_pol = (data["pols"][..., None]) # (B, Nevs, 1)
        est_C_thres = None
        if self.log_implicit_C_thres:
            est_C_thres = estimate_C_thres_from_pol_dL(gt_pol, delta_linlog, esim=True) # for debugging

        # Compute Loss
        if (self.C_thres != -1):
            # sometimes torch.abs() looks better than **2, e.g. for rgb+events on Shake Carpet 1
            loss_evs = w_evLoss * torch.mean((delta_linlog - gt_pol * self.C_thres)**2) 
        else:
            EPS = 1e-9
            w_evLoss *= 20 # larger weight for better comparability
            if not self.event_only:
                raise NotImplementedError('only support event_only=True')
                # w_evLoss *= 20 # seems to help for normalized loss
            delta_linlog_normed = delta_linlog / (torch.linalg.norm(delta_linlog, dim=1, keepdim=True) + EPS)
            sum_pol_normed = gt_pol / (torch.linalg.norm(gt_pol, dim=1, keepdim=True) + EPS)
            loss_evs = w_evLoss * torch.mean((delta_linlog_normed - sum_pol_normed)**2)
        
        loss = loss_evs
        if not self.event_only:
            # rays_o = data['rays_o'] # [B, N, 3]
            # rays_d = data['rays_d'] # [B, N, 3]

            # # train with random background color if using alpha mixing
            # if C == 4:
            #     bg_color = torch.rand_like(images[..., :self.out_dim_color]) # [B, N, 3], pixel-wise random.
            #     gt_rgb = images[..., :self.out_dim_color] * images[..., self.out_dim_color:] + bg_color * (1 - images[..., self.out_dim_color:])
            # else:
            #     bg_color = None
            #     gt_rgb = images

            # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, **vars(self.opt))
            # pred_rgb = outputs['image']
            # loss_frames = self.criterion(pred_rgb, gt_rgb).mean()
            # loss = loss + self.weight_loss_rgb * loss_frames
            raise NotImplementedError('only support event_only=True')

        if self.negative_event_sampling and self.epoch > self.epoch_start_noEvLoss:
            # bg_color_evs = torch.rand((B, 1, self.out_dim_color)).to(self.device) # (B, Nnoevs, 3)
            # outputs1 = self.model.render(data["rays_no_evs_o1"], data["rays_no_evs_d1"], staged=False, bg_color=bg_color_evs, perturb=True, **vars(self.opt))
            # outputs2 = self.model.render(data["rays_no_evs_o2"], data["rays_no_evs_d2"], staged=False, bg_color=bg_color_evs, perturb=True, **vars(self.opt))

            # # Convert I => log(I)
            # if self.use_luma:
            #     pred_luma1 = rgb_to_luma(outputs1["image"], esim=True) # (B, Nnoevs, 1)
            #     pred_linlog1 = lin_log(pred_luma1*255, linlog_thres=20) # (B, Nnoevs, 1)

            #     pred_luma2 = rgb_to_luma(outputs2["image"], esim=True) # (B, Nnoevs, 1)
            #     pred_linlog2 = lin_log(pred_luma2*255, linlog_thres=20) # (B, Nnoevs, 1)
            # else:
            #     pred_linlog1 = lin_log(outputs1["image"]*255, linlog_thres=20) # (B, Nnoevs, 3)
            #     pred_linlog2 = lin_log(outputs2["image"]*255, linlog_thres=20) # (B, Nnoevs, 3)

            # abs_diff = torch.abs(pred_linlog2 - pred_linlog1) # (B, Nnoevs, 1or3)
            # Cno = self.C_thres if self.C_thres > 0 else 0.25 
            # loss_no_evs = self.w_no_ev * torch.mean(torch.relu(abs_diff - Cno))
            # loss = loss + loss_no_evs
            raise NotImplementedError('negative sampling not supported yet')

        losses = {}
        losses["loss_evs"] = loss_evs
        losses["loss_no_evs"] = loss_no_evs
        losses["loss_frames"] = loss_frames
        return delta_linlog, gt_pol, loss, est_C_thres, losses
    
    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
            'model.out_dim_color': self.model.out_dim_color,
            'model.raw_poses_hf': self.model.raw_poses_hf,
            'model.raw_tss_poses_hf_ns': self.model.raw_tss_poses_hf_ns
        }

        if self.model.nerf.cuda_ray:
            state['mean_count'] = self.model.nerf.mean_count
            state['mean_density'] = self.model.nerf.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['optimizer_pose'] = self.optimizer_pose.state_dict()
            state['lr_scheduler_pose'] = self.lr_scheduler_pose.state_dict()
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
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

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
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.optimizer_pose and  'optimizer_pose' in checkpoint_dict:
            try:
                self.optimizer_pose.load_state_dict(checkpoint_dict['optimizer_pose'])
                self.log("[INFO] loaded optimizer_pose.")
            except:
                self.log("[WARN] Failed to load optimizer_pose.")
        
        if self.lr_scheduler_pose and 'lr_scheduler_pose' in checkpoint_dict:
            try:
                self.lr_scheduler_pose.load_state_dict(checkpoint_dict['lr_scheduler_pose'])
                self.log("[INFO] loaded scheduler_poses.")
            except:
                self.log("[WARN] Failed to load scheduler_poses.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def evaluate_one_epoch(self, loader, name=None):
        assert self.event_only, "only event_only=True supported."
        assert not self.eval_stereo_views, "we don't support evaluate the event camera's view generation."
        super().evaluate_one_epoch(loader, name=name)

        # * output the refined poses_hf based on current model parameters for visualization
        save_poses_hf_ref_path = os.path.join(self.workspace, "validation", "poses_hf_ref_raw")
        os.makedirs(save_poses_hf_ref_path, exist_ok=True)
        with torch.no_grad():
            poses_hf_ref = self.model.compute_refined_poses_hf().detach().cpu()
        evs_timespan_us = torch.tensor(self.evs_timespan_us) if hasattr(self, 'evs_timespan_us') else None
        poses_hf_dict = {'poses_hf': self.model.poses_hf.detach().cpu(),
                         'poses_hf_ref': poses_hf_ref,
                         'tss_poses_hf_ns': self.model.tss_poses_hf_ns.detach().cpu(),
                         'raw_poses_hf': self.model.raw_poses_hf.detach().cpu(),
                         'raw_tss_poses_hf_ns': self.model.raw_tss_poses_hf_ns.detach().cpu(),
                         'evs_timespan_us': evs_timespan_us,
                        #  'poses_hf_dict': self.poses_hf_dict_final,
                         'epoch': self.epoch}
        # # todo new interface, enable later (check if the two implementations have the same value before change)
        # poses_hf_dict = {'poses_hf_ref': poses_hf_ref,
        #                  'evs_timespan_us': evs_timespan_us,
        #                  'poses_hf_dict': self.poses_hf_dict_final,
        #                  'epoch': self.epoch}
        prefix = '' if name is None else name+'_'
        with open(os.path.join(save_poses_hf_ref_path, f'{prefix}poses_hf_ref_{self.epoch:08d}.pickle'), 'wb') as fout:
            pickle.dump(poses_hf_dict, fout)

    def eval_step_tumvie(self, data, loader):
        raise NotImplemented("we don't support evaluate the event camera's view generation.")

    def eval_step(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]

        B, H, W, C = images.shape
        # C = self.out_dim_color

        if self.opt.color_space == 'linear':
            images[..., :self.out_dim_color] = srgb_to_linear(images[..., :self.out_dim_color])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :self.out_dim_color] * images[..., self.out_dim_color:] + bg_color * (1 - images[..., self.out_dim_color:])
        else:
            gt_rgb = images
        
        outputs = self.model.nerf.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, self.out_dim_color)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss
    
    @classmethod
    def get_optimizer_and_scheduler(cls, lr, lr_pose, total_iters, opt):
        optimizer, scheduler = cls._get_optimizer_and_scheduler_for_nerf(lr, total_iters)
        optimizer_pose, scheduler_pose = cls._get_optimizer_and_scheduler_for_pose(lr_pose, total_iters)
        scheduler_update_every_step, scheduler_pose_update_every_step = True, True
        return optimizer, scheduler, optimizer_pose, scheduler_pose, scheduler_update_every_step, scheduler_pose_update_every_step
    
    @classmethod
    def _get_optimizer_and_scheduler_for_nerf(cls, lr, total_iters):
        optimizer = lambda model: torch.optim.Adam(model.nerf.get_params(lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / total_iters, 1))
        return optimizer, scheduler
    
    @classmethod
    def _get_optimizer_and_lambda_lr_scheduler_for_pose(cls, lr_pose, total_iters):
        optimizer_pose = lambda model: torch.optim.Adam(model.se3_refine.parameters(), lr=lr_pose)
        scheduler_pose = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / total_iters, 1))
        return optimizer_pose, scheduler_pose

    _get_optimizer_and_scheduler_for_pose = _get_optimizer_and_lambda_lr_scheduler_for_pose

    # todo
    def test(self, loader, save_path=None, name=None):
        pass

    def test_step(self, data, bg_color=None, perturb=False):
        pass

    def save_mesh(self, save_path=None, resolution=256, threshold=10):
        pass



    # deprecated methods
    def train_gui(self, train_loader, step=16):
        raise NotImplementedError
    
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        raise NotImplementedError
    
    def evaluate(self, loader, name=None):
        raise NotImplementedError
    
    def train_step(self, data):
        raise NotImplementedError