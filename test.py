import argparse
import time
import numpy as np
import PIL.Image as Image
from datasets.dataset import Crowd
import torch
import os
from torch.utils.data import DataLoader
from model.model_vimoc import vgg19
import torch.nn.functional as F

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--label_info', default='label_list/VIMOC-1F.txt',
                        help="assign device")
    parser.add_argument('--data-dir', default='/data/macz/MRC-Crowd-main/processed_VIMOC2',
                        help='the directory of the data')
    parser.add_argument('--model-path', default='',
                        help='the path to the model')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--device', default='1',
                        help="assign device")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    # dataset = Crowd(args.data_dir, 512, args.downsample_ratio, method='val')
    dataset = {x: Crowd(os.path.join(args.data_dir, x),
                        crop_size=512,
                        downsample_ratio=args.downsample_ratio,
                        info=args.label_info,
                        method=x) for x in ['train', 'test']}
    val_loader = DataLoader(dataset['test'], batch_size=1, shuffle=False)
    # dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=False)
    model = vgg19(25)
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    model.eval()
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    file = open('result2.txt', 'w')
    step = 0
    epoch_res = []

    for inputs_prev, inputs, inputs_post, gt_counts, im_path in val_loader:
         
            inputs = inputs.to(device)
            inputs_prev = inputs_prev.to(device)
            with torch.set_grad_enabled(False):
                B, C, H, W = inputs.size()

                prev_flow, u_t_cls, _ = model(inputs_prev, inputs) 
        
                mask_boundry = torch.zeros(prev_flow.shape[2:])
                mask_boundry[0,:] = 1.0
                mask_boundry[-1,:] = 1.0
                mask_boundry[:,0] = 1.0
                mask_boundry[:,-1] = 1.0
                mask_boundry = mask_boundry.cuda()

                output = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

                res = gt_counts[0].item() - torch.sum(output).item()
                
                epoch_res.append(res)
                # print('{}/{} {}: predict:{}, gt:{}, diff:{}'.format(step, len(dataset), name, torch.sum(output).item(), gt_counts[0].item(), res),  np.abs(np.array(epoch_res)).mean())
                step += 1
                #file.write('{}/{} {} {} {}\n'.format(name, len(dataset), torch.sum(output).item(), gt_counts[0].item(), res))
    epoch_res = np.abs(np.array(epoch_res))
 
    print('MAE:{:.2f}, MSE:{:.2f}'.format(epoch_res.mean(), np.sqrt((epoch_res**2).mean())))
 
 