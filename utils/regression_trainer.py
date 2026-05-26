from torch.utils.data import DataLoader
import torch
import logging
from utils.helper import SaveHandler, AverageMeter
from utils.trainer import Trainer
from model.model_vimoc import vgg19,vgg16
from datasets.dataset import Crowd, TwoStreamBatchSampler
import numpy as np
import os
import time
import random
import torch.nn.functional as F
import torch.nn as nn
from loss.seg_loss import SegmentationLoss
from loss.ssim_loss import cal_avg_ms_ssim
from utils.den_cls import den2cls
from utils.mask_geneator import MaskGenerator, repeat_fun


def get_normalized_map(density_map):
    B, C, H, W = density_map.size()
    #mu_sum = density_map.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_sum = density_map.reshape([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)

    mu_normed = density_map / (mu_sum + 1e-6)
    return mu_normed


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def kd_density_loss_log(y_s, y_t, eps=1e-6):
    # Clamp 保证输入范围在 [eps, +∞)
    y = torch.clamp(y_t, min=eps)
    y_pred = torch.clamp(y_s, min=eps)

    #loss = y * abs((torch.log(y) - torch.log(y_pred)))
    loss = y * torch.abs(torch.log(y) - torch.log(y_pred))
    return loss 
    
class Reg_Trainer(Trainer):
    def setup(self):
        args = self.args
        if args.seed != -1:
            setup_seed(args.seed)
            print('Random seed is set as {}'.format(args.seed))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(torch.cuda.current_device())
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('Using {} gpus'.format(self.device_count))
        else:
            raise Exception('GPU is not available')

        self.d_ratio = args.downsample_ratio

        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  crop_size=args.crop_size,
                                  downsample_ratio=self.d_ratio,
                                  info=args.label_info,
                                  method=x) for x in ['train', 'test']}

        logging.info('Number of images in the dataset {}, with {} labeled data and {} unlabeled data'.format(len(self.datasets['train']),
                                                                                                             len(self.datasets['train'].labeled_idx),
                                                                                                             len(self.datasets['train'].unlabeled_idx)))

        train_sampler = TwoStreamBatchSampler(self.datasets['train'].unlabeled_idx, self.datasets['train'].labeled_idx,
                                              args.batch_size, args.num_labeled)
        self.train_loader = DataLoader(self.datasets['train'], batch_sampler=train_sampler)
        self.val_loader = DataLoader(self.datasets['test'], batch_size=1, shuffle=False)

        self.label_count = torch.tensor(
            [0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765,
             0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087,
             0.17838051915168762, 0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755,
             0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993,
             1.5706151723861694, 2.138580322265625, 3.233219861984253, 7.914860725402832]).to(self.device)

        self.model = vgg19(len(self.label_count) + 1)
        self.model.to(self.device)
        self.ema_model = vgg19(len(self.label_count) + 1)
        self.ema_model.to(self.device)
        self.mask_generator = MaskGenerator(args.crop_size, args.mask_size, args.downsample_ratio, args.mask_ratio)
        self.cls_loss = SegmentationLoss().to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        self.global_step = 0
        self.ramup = exp_rampup(args.weight_ramup)
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step']
            elif suf == 'pth':
                raise Exception('Not supported')

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_list = SaveHandler(num=args.max_num)

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-' * 50 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 50)
            self.epoch = epoch
            self.train_epoch()
            if self.epoch >= args.start_val and self.epoch % args.val_epoch == 0:
                self.val_epoch()

    def train_epoch(self):
        epoch_reg_loss = AverageMeter()
        epoch_cls_loss = AverageMeter()
        epoch_unsupervised_loss =AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()
        self.ema_model.train()

        for step, (png, w_inputs_prev, w_inputs, w_inputs_post, s_inputs_prev, s_inputs, s_inputs_post, den_map, labels) in enumerate(
                self.train_loader):
            png = png.to(self.device)
            w_inputs = w_inputs.to(self.device)
            w_inputs_prev = w_inputs_prev.to(self.device)
            w_inputs_post = w_inputs_post.to(self.device)
            s_inputs = s_inputs.to(self.device)
            s_inputs_prev = s_inputs_prev.to(self.device)
            s_inputs_post = s_inputs_post.to(self.device)
            gt_den_map = den_map.to(self.device)

            with torch.set_grad_enabled(True):
                self.global_step += 1
                N = w_inputs.size(0)
                N_l = w_inputs[labels].size(0)
                N_u = w_inputs.size(0) - N_l

                N2 = w_inputs_prev.size(0)
                N_l2 = w_inputs_prev[labels].size(0)
                N_u2 = w_inputs_prev.size(0) - N_l2
             
                prev_flow, cls_score,depth = self.model(w_inputs_prev[labels], w_inputs[labels])

          
             #   prev_flow_inverse, _ = self.model(w_inputs[labels], w_inputs_prev[labels])

                mask_boundry = torch.zeros(prev_flow.shape[2:])
                mask_boundry[0,:] = 1.0
                mask_boundry[-1,:] = 1.0
                mask_boundry[:,0] = 1.0
                mask_boundry[:,-1] = 1.0
                mask_boundry = mask_boundry.cuda()

                pred = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
             #   pred2 = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
             #   reconstruction_from_post_inverse = F.pad(post_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0))+post_flow_inverse[0,4,:,:]+F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+post_flow_inverse[0,9,:,:]*mask_boundry
             
                # loss_prev_consistency = get_reg_loss(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1]) + \
                #         get_reg_loss(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:]) + \
                #         get_reg_loss(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:]) + \
                #         get_reg_loss(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1]) + \
                #         get_reg_loss(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:]) + \
                #         get_reg_loss(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:]) + \
                #         get_reg_loss(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1]) + \
                #         get_reg_loss(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:]) + \
                #         get_reg_loss(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])
    
                ################Supervised Loss###############  
                reg_loss = get_reg_loss(pred, gt_den_map[labels])
                depth_loss = get_reg_loss(depth, png[labels])    
               
                epoch_reg_loss.update(reg_loss.item(), N_l)
                gt_cls_map = den2cls(gt_den_map, self.label_count)
                cls_loss = self.cls_loss(cls_score, gt_cls_map[labels]).mean()
                epoch_cls_loss.update(cls_loss.item(), N_l)
                
                loss = reg_loss + 0.01 * depth_loss  + cls_loss
                ###############UnSupervised Loss#################
                self.update_ema_model(self.model, self.ema_model, self.args.ema_decay, self.global_step)
                masks = repeat_fun(N_u, self.mask_generator).to(self.device)
                
                masks = masks.unsqueeze(1)
                if masks.dtype != torch.float32:
                    masks = masks.float()

                masks = F.interpolate(masks, size=(48, 80), mode='bilinear', align_corners=False)
                 
                masks = masks.squeeze(1)
                input_mask = 1 - masks.repeat_interleave(8, 1).repeat_interleave(8, 2).unsqueeze(1).contiguous() # masked area=0, unmasked = 1
                input_mask2 = masks.repeat_interleave(8, 1).repeat_interleave(8, 2).unsqueeze(1).contiguous()  # masked area=1, unmasked = 0
               
                masks = masks.unsqueeze(1) # masked area =1, unmasked = 0
                masks2 = 1 - masks
                # if input_mask.dtype != torch.float32:
                #     input_mask = input_mask.float()

                # input_mask = F.interpolate(input_mask, size=(360, 640), mode='bilinear', align_corners=False)
                
                prev_flow, u_s_cls,u_s_depth = self.model(s_inputs_prev[labels == 0] * input_mask, s_inputs[labels == 0] * input_mask)
                prev_flow2, u_s_cls2,u_s_depth2 = self.model(s_inputs_prev[labels == 0] * input_mask2, s_inputs[labels == 0] * input_mask2)
              #  prev_flow_inverse, _ = self.model(s_inputs[labels == 0] * input_mask, s_inputs_prev[labels == 0] * input_mask)

                prev_flow_inverse, _,_ = self.model(s_inputs[labels == 0] * input_mask, s_inputs_post[labels == 0] * input_mask)
                prev_flow_inverse2, _,_ = self.model(s_inputs[labels == 0] * input_mask2, s_inputs_post[labels == 0] * input_mask2)
                u_s_reg = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
                u_s_reg2 = F.pad(prev_flow2[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow2[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow2[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow2[0,3,:,1:],(0,1,0,0))+prev_flow2[0,4,:,:]+F.pad(prev_flow2[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow2[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow2[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow2[0,8,:-1,:-1],(1,0,1,0))+prev_flow2[0,9,:,:]*mask_boundry
                u_s_reg_i2 = torch.sum(prev_flow_inverse2[0,:9,:,:],dim=0)+prev_flow_inverse2[0,9,:,:]*mask_boundry
                u_s_reg_i = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
                with torch.no_grad():
                    prev_flow, u_t_cls,u_t_depth = self.ema_model(w_inputs_prev[labels == 0], w_inputs[labels == 0])   
                #    prev_flow_inverse, _ = self.ema_model(w_inputs[labels == 0], w_inputs_prev[labels == 0]) 
                    prev_flow_inverse, _,_ = self.ema_model(w_inputs[labels == 0], w_inputs_post[labels == 0]) 

                    u_t_reg = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry   
                    u_t_reg2 = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
                    
                    u_t_cls = u_t_cls.detach()
                    u_t_reg = u_t_reg.detach()   
                    u_t_reg2 = u_t_reg2.detach() 
                    u_t_depth = u_t_depth.detach() 
                u_mreg_loss = (nn.L1Loss(reduction='none')(u_s_reg, u_t_reg) * masks).sum() / (masks.sum() + 1e-5)
                u_mreg_loss2 = (nn.L1Loss(reduction='none')(u_s_reg2, u_t_reg) * masks2).sum() / (masks.sum() + 1e-5)
                u_mreg_loss3 = (nn.L1Loss(reduction='none')(u_s_reg_i, u_t_reg2)* masks).sum() / (masks.sum() + 1e-5)
                u_mreg_loss4 = (nn.L1Loss(reduction='none')(u_s_reg_i2, u_t_reg2)* masks2).sum() / (masks.sum() + 1e-5)

                u_mcls_loss = (nn.L1Loss(reduction='none')(u_s_cls.softmax(dim=1), u_t_cls.softmax(dim=1)) * masks).sum() / (masks.sum() + 1e-5)
                u_mcls_loss2 = (nn.L1Loss(reduction='none')(u_s_cls2.softmax(dim=1), u_t_cls.softmax(dim=1)) * masks2).sum() / (masks2.sum() + 1e-5)
                u_depth_loss = (nn.L1Loss(reduction='none')(u_s_depth.softmax(dim=1), u_t_depth.softmax(dim=1)) * masks).sum() / (masks.sum() + 1e-5)
                u_depth_loss2 = (nn.L1Loss(reduction='none')(u_s_depth2.softmax(dim=1), u_t_depth.softmax(dim=1)) * masks2).sum() / (masks2.sum() + 1e-5)

                u_kd_loss = (kd_density_loss_log(u_s_reg, u_t_reg)* masks).sum() / (masks.sum() + 1e-5)
                u_kd_loss2 = (kd_density_loss_log(u_s_reg2, u_t_reg)* masks2).sum() / (masks.sum() + 1e-5)
                u_kd_loss3 = (kd_density_loss_log(u_s_reg_i, u_t_reg2)* masks).sum() / (masks2.sum() + 1e-5)
                u_kd_loss4 = (kd_density_loss_log(u_s_reg_i2, u_t_reg2)* masks2).sum() / (masks2.sum() + 1e-5)

                u_self_loss = nn.L1Loss()(u_t_reg, u_t_reg2)

                cons_loss =  u_mcls_loss + u_mcls_loss2 + u_mreg_loss + u_mreg_loss2 + u_mreg_loss3 + u_mreg_loss4 + u_self_loss  + 0.01*(u_depth_loss+u_depth_loss2)
              
                epoch_unsupervised_loss.update(cons_loss.item(), N_u)
                # #################################
                loss += 1.0 * cons_loss * self.ramup(self.epoch)

                gt_counts = torch.sum(gt_den_map[labels].view(N_l, -1), dim=1).detach().cpu().numpy()
                pred_counts = torch.sum(pred.view(N_l, -1), dim=1).detach().cpu().numpy()
                # print(gt_counts)
                # print(pred_counts)
                diff = pred_counts - gt_counts
                epoch_mae.update(np.mean(np.abs(diff)).item(), N)
                epoch_mse.update(np.mean(diff * diff), N)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        logging.info(
            'Epoch {} Train, reg:{:.4f}, cls_score:{:.4f}, unsupervised:{:.4f} mae:{:.2f}, mse:{:.2f}, Cost: {:.1f} sec '
            .format(self.epoch, epoch_reg_loss.getAvg(), epoch_cls_loss.getAvg(), epoch_unsupervised_loss.getAvg(), epoch_mae.getAvg(),
                    np.sqrt(epoch_mse.getAvg()), (time.time() - epoch_start)))

        if self.epoch % 5 == 0:
            model_state_dict = self.model.state_dict()
            ema_model_state_dict = self.ema_model.state_dict()
            save_path = os.path.join(self.save_dir, "{}_ckpt.tar".format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dict,
                'ema_model_state_dict': ema_model_state_dict,
                'global_step': self.global_step
            }, save_path)
            self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []
        for inputs_prev, inputs, inputs_post, gt_counts, name in self.val_loader:
          
            inputs = inputs.to(self.device)
            inputs_prev = inputs_prev.to(self.device)
            with torch.set_grad_enabled(False):
                prev_flow, u_t_cls, depth = self.model(inputs_prev, inputs) 

                mask_boundry = torch.zeros(prev_flow.shape[2:])
                mask_boundry[0,:] = 1.0
                mask_boundry[-1,:] = 1.0
                mask_boundry[:,0] = 1.0
                mask_boundry[:,-1] = 1.0
                mask_boundry = mask_boundry.cuda()

                outputs = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
               # outputs2 = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
                 
                res = gt_counts[0].item() - torch.sum(outputs).item()
                
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Val, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))

        model_state_dict = self.model.state_dict()

        if (mae + mse) < (self.best_mae + self.best_mse):
            self.best_mae = mae
            self.best_mse = mse
            torch.save(model_state_dict, os.path.join(self.save_dir, 'best_model.pth'.format(self.epoch)))
            logging.info("Save best model: MAE: {:.2f} MSE:{:.2f} model epoch {}".format(mae, mse, self.epoch))
        print("Best Result: MAE: {:.2f} MSE:{:.2f}".format(self.best_mae, self.best_mse))

    def update_ema_model(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)



def get_reg_loss(pred, gt, threshold=1e-3, level=3, window_size=3):
    mask = gt > threshold
 
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]     
    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
  
    loss_ssim = cal_avg_ms_ssim(pred * mask, gt * mask, level=level,
                                window_size=window_size)
    mu_normed = get_normalized_map(pred)
    gt_mu_normed = get_normalized_map(gt)
    tv_loss = (nn.L1Loss(reduction='none')(mu_normed, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
    return loss_ssim + 0.01 * tv_loss


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper
