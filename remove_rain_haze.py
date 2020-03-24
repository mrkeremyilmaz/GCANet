import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from .utils import make_dataset, edge_compute

class ModelProp:
    def __init__(self, task):
        self.network = 'GCANet'
        self.task = task
        self.gpu_id = 0 
        self.only_residual = self.task == 'dehaze'  
        self.model = '/home/keremyilmazcs/general_photonom_server/photonom-django/photonom/image_operator/operations' \
                     '/GCANet/models/wacv_gcanet_%s.pth' % self.task
        self.use_cuda = self.gpu_id >= 0

def load_model(task):
    opt = ModelProp(task)
    
    if opt.network == 'GCANet':
        from .GCANet import GCANet
        net = GCANet(in_c=4, out_c=3, only_residual=opt.only_residual)
    else:
        print('network structure %s not supported' % opt.network)
        raise ValueError

    if opt.use_cuda:
        torch.cuda.set_device(opt.gpu_id)
        net.cuda()
    else:
        net.float()
    
    net.load_state_dict(torch.load(opt.model, map_location='cpu'))
    net.eval()
    return net, opt
    

def remove_rain_haze(img_path, task, net, opt):
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    edge_data = edge_compute(img_data)
    in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128 
    in_data = in_data.cuda() if opt.use_cuda else in_data.float()
    with torch.no_grad():
        pred = net(Variable(in_data))
    if opt.only_residual:
        out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)
    else:
        out_img_data = pred.data[0].cpu().float().round().clamp(0, 255)
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    return out_img

img_path = "D:\Bilkent Fall-2019\CS491\RainRemoval\GCANet\input\haze1.jpeg"
h_net, h_opt = load_model("dehaze")
r_net, r_opt = load_model("derain")
out_img = remove_rain_haze(img_path, "dehaze", h_net, h_opt)
out_img.show()
out_img = remove_rain_haze(img_path, "derain", r_net, r_opt)
out_img.show()
    





