import torch
import numpy as np
import cv2
import sys
import os
import tempfile
import shutil
import importlib
from easydict import EasyDict


class LossLogger():
    def __init__(self, recorder, data_split='train'):
        self.recorder = recorder
        self.data_split = data_split
        self.reset()

    def reset(self):
        self.loss_stats = {} # each will be 
    
    def update(self, loss_dict):
        for key in loss_dict:
            if key not in self.loss_stats:
                self.loss_stats[key] = AverageMeter()
            self.loss_stats[key].update(loss_dict[key].mean().item())
    
    def log(self, step):
        for key in self.loss_stats:
            name = key + '/' + self.data_split
            self.recorder.add_scalar(name, self.loss_stats[key].avg, step)

def convertAlpha2Rot(alpha, cx, P2):
    cx_p2 = P2[..., 0, 2]
    fx_p2 = P2[..., 0, 0]
    ry3d = alpha + np.arctan2(cx - cx_p2, fx_p2)
    ry3d[np.where(ry3d > np.pi)] -= 2 * np.pi
    ry3d[np.where(ry3d <= -np.pi)] += 2 * np.pi
    return ry3d


def convertRot2Alpha(ry3d, cx, P2):
    cx_p2 = P2[..., 0, 2]
    fx_p2 = P2[..., 0, 0]
    alpha = ry3d - np.arctan2(cx - cx_p2, fx_p2)
    alpha[alpha > np.pi] -= 2 * np.pi
    alpha[alpha <= -np.pi] += 2 * np.pi
    return alpha

def alpha2theta_3d(alpha, x, z, P2):
    """ Convert alpha to theta with 3D position
    Args:
        alpha [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        theta []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(alpha, torch.Tensor):
        theta = alpha + torch.atan2(x + offset, z)
    else:
        theta = alpha + np.arctan2(x + offset, z)
    return theta

def theta2alpha_3d(theta, x, z, P2):
    """ Convert theta to alpha with 3D position
    Args:
        theta [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        alpha []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(theta, torch.Tensor):
        alpha = theta - torch.atan2(x + offset, z)
    else:
        alpha = theta - np.arctan2(x + offset, z)
    return alpha

def draw_3D_box(img, corners, color = (255, 255, 0)):
    """
        draw 3D box in image with OpenCV,
        the order of the corners should be the same with BBox3dProjector
    """
    points = np.array(corners[0:2], dtype=np.int32) #[2, 8]
    points = [tuple(points[:,i]) for i in range(8)]
    for i in range(1, 5):
        cv2.line(img, points[i], points[(i%4+1)], color, 2)
        cv2.line(img, points[(i + 4)%8], points[((i)%4 + 5)%8], color, 2)
    cv2.line(img, points[2], points[7], color)
    cv2.line(img, points[3], points[6], color)
    cv2.line(img, points[4], points[5],color)
    cv2.line(img, points[0], points[1], color)
    return img

def compound_annotation(labels, max_length, bbox2d, bbox_3d, obj_types):
    """ Compound numpy-like annotation formats. Borrow from Retina-Net
    
    Args:
        labels: List[List[str]]
        max_length: int, max_num_objects, can be dynamic for each iterations
        bbox_2d: List[np.ndArray], [left, top, right, bottom].
        bbox_3d: List[np.ndArray], [cam_x, cam_y, z, w, h, l, alpha].
        obj_types: List[str]
    Return:
        np.ndArray, [batch_size, max_length, 12]
            [x1, y1, x2, y2, cls_index, cx, cy, z, w, h, l, alpha]
            cls_index = -1 if empty
    """
    annotations = np.ones([len(labels), max_length, bbox_3d[0].shape[-1] + 5]) * -1
    for i in range(len(labels)):
        label = labels[i]
        for j in range(len(label)):
            annotations[i, j] = np.concatenate([
                bbox2d[i][j], [obj_types.index(label[j])], bbox_3d[i][j]
            ])
    return annotations

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.best = sys.maxsize

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.best = min(self.best, self.avg)

def cfg_from_file(cfg_filename:str)->EasyDict:
    assert cfg_filename.endswith('.py')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(cfg_filename, os.path.join(temp_config_dir, temp_config_name))
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        cfg = getattr(importlib.import_module(temp_module_name), 'cfg')
        assert isinstance(cfg, EasyDict)
        sys.path.pop()
        del sys.modules[temp_module_name]
        temp_config_file.close()

    return cfg




def save_model(path_save, epoch, loss, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()


    data = {'epoch': epoch, 'loss':loss, 'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path_save)
    print('New model was saved to', path_save)



def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=[20, 40]):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
  
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the '+\
            'pre-trained weight. Please make sure '+\
            'you have correctly specified --arch xxx '+\
            'or set the correct --num_classes for your own dataset.'

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                    'loaded shape{}. {}'.format(
                k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # start_epoch = checkpoint['epoch']
            start_lr = lr
            # for step in lr_step:
            #     if start_epoch >= step:
            #         start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, checkpoint['epoch'], checkpoint['loss']
    else:
        return model