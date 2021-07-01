import numpy as np
import os
import pickle
import time
from PIL import Image
from copy import deepcopy
import skimage.measure
import torch

from _path_init import *
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData, NuscenesData
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file

import json
from itertools import groupby
import threading



# global variable x
count = -1

def increment():
    global count
    idx = count + 1
    count += 1
    return idx



def thread_task(thread_id, limit, cfg, data_split, 
anchor_prior, nusc, sample_data, save_dir, total_objects, total_usable_objects,
uniform_sum_each_type, uniform_square_each_type, preprocess, examine, sums, squared):

    

    anchor_manager = Anchors(cfg.path.preprocessed_path, readConfigFile=False, **cfg.detector.head.anchors_cfg)
    
    start_idx, end_idx = limit
    frames = [None] * (end_idx - start_idx)

    time_display_inter = 200
    data_root_dir = cfg.path.data_path
    # lock.acquire()
    # idx = increment()
    # lock.release()

    old_idx = -1
    # while idx < limit:
    for internal_idx, idx in enumerate(range(start_idx, end_idx)):
        # assert old_idx != idx, 'None data frame'
        if old_idx >= 0:
            assert frames[old_idx] is not None, f'data is None!! thread: {thread_id}, sample {old_idx}'

        old_idx = internal_idx
        
    
        # assert checker[idx] == 0, 'wrong in threading!'
        # checker[idx] = thread_id
        

        # read data with dataloader api
        # start_t = time.time()

        image_id, anno_data = sample_data[idx]
        data_frame = NuscenesData(
            data_root_dir, 
            nusc['images'][image_id], anno_data, nusc['categories'],
            preprocess_phase=True
        )
        # print(time.time() - start_t, end=' - ')

        # start_t = time.time()
        calib, image_pth, label = data_frame.read_data()
        # print(time.time() - start_t)
        # store the list of kittiObjet and kittiCalib
        max_occlusion = getattr(cfg.data, 'max_occlusion', 2)
        min_z        = getattr(cfg.data, 'min_z', 3)

        if data_split == 'training':
            data_frame.label = [
                obj 
                for obj in label.data 
                if obj.type in cfg.obj_types and obj.occluded < max_occlusion and obj.z > min_z
            ]
            
            if anchor_prior:
                for j in range(len(cfg.obj_types)):
                    total_objects[j] += len([1 for obj in data_frame.label if obj.type==cfg.obj_types[j]])
            
                    data = np.array([
                        [obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha), obj.w, obj.h, obj.l] 
                        for obj in data_frame.label 
                        if obj.type==cfg.obj_types[j]
                    ]) #[N, 6]
                    if data.any():
                        uniform_sum_each_type[j, :] += np.sum(data, axis=0)
                        uniform_square_each_type[j, :] += np.sum(data ** 2, axis=0)
        else:
            data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types]
        
        data_frame.calib = calib
        
        if data_split == 'training' and anchor_prior:
            image = np.array(Image.open(image_pth, 'r'))
            original_image = image.copy()
            # baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0]
            image, P2, label = preprocess(
                original_image, p2=deepcopy(calib.P2), labels=deepcopy(data_frame.label)
            )
            # _,  P3 = preprocess(original_image, p2=deepcopy(calib.P3))

            ## Computing statistic for positive anchors
            if len(data_frame.label) > 0:
                anchors, _ = anchor_manager(
                    image[np.newaxis].transpose([0,3,1,2]), 
                    torch.tensor(P2).reshape([-1, 3, 4])
                )

                for j in range(len(cfg.obj_types)):
                    bbox2d = torch.tensor([
                        [obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] 
                        for obj in label 
                        if obj.type == cfg.obj_types[j]
                    ]).cuda()
                    if len(bbox2d) < 1:
                        continue

                    bbox3d = torch.tensor([[
                        obj.x, obj.y, obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha)] 
                        for obj in label 
                        if obj.type == cfg.obj_types[j]
                    ]).cuda()

                    
                    usable_anchors = anchors[0]

                    IoUs = calc_iou(usable_anchors, bbox2d) #[N, K]
                    IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                    IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                    num_usable_object = torch.sum(IoU_max > cfg.detector.head.loss_cfg.fg_iou_threshold).item()
                    total_usable_objects[j] += num_usable_object

                    positive_anchors_mask = IoU_max_anchor > cfg.detector.head.loss_cfg.fg_iou_threshold
                    positive_ground_truth_3d = bbox3d[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()

                    used_anchors = usable_anchors[positive_anchors_mask].cpu().numpy() #[x1, y1, x2, y2]

                    sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)

                    for k in range(len(sizes_int)):
                        examine[j, sizes_int[k], ratio_int[k]] += 1
                        sums[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                        squared[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2
        
        frames[internal_idx] = data_frame
        
        
        if internal_idx % time_display_inter == 0:
            print(f"{data_split} -- {thread_id} iter:{internal_idx}, total_objs:{total_objects}, usable_objs:{total_usable_objects}")


        # lock.acquire()
        # idx = increment()
        # lock.release()

    pkl_file = os.path.join(save_dir, f'imdb_{thread_id}.pkl')
    pickle.dump(frames, open(pkl_file, 'wb'))

    print(f'Thread {thread_id} done!!!')


def read_one_split(cfg, data_split='training'):

    anno_pth = cfg.data.train_split_file if data_split=='training' else cfg.data.val_split_file
    with open(anno_pth) as f:
        nusc = json.load(f)
    
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    

    anchor_prior = getattr(cfg, 'anchor_prior', True)

    num_thread = 10
    total_objects = [[0 for _ in range(len(cfg.obj_types))]] * num_thread
    total_usable_objects = [[0 for _ in range(len(cfg.obj_types))]] * num_thread
    


    if anchor_prior:
        preprocess = build_augmentator(cfg.data.test_augmentation)
        
        
        len_scale = len(cfg.detector.head.anchors_cfg.scales)
        len_ratios = len(cfg.detector.head.anchors_cfg.ratios)
        len_level = len(cfg.detector.head.anchors_cfg.pyramid_levels)

        examine = [np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios])] * num_thread
        sums    = [np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3])] * num_thread
        squared = [np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3], dtype=np.float64)] * num_thread

        uniform_sum_each_type = [np.zeros((len(cfg.obj_types), 6), dtype=np.float64)] * num_thread #[z, sin2a, cos2a, w, h, l]
        uniform_square_each_type = [np.zeros((len(cfg.obj_types), 6), dtype=np.float64)] * num_thread


    nusc['categories'] = {d['id']: d for d in nusc['categories']}
    nusc['images'] = {d['id']: d for d in nusc['images']}
    
    # sort INFO data by 'company' key.
    nusc['annotations'] = sorted(nusc['annotations'], key=lambda k: k['image_id'])
    # sample_data = groupby(nusc['annotations'], lambda k: k['image_id'])
    sample_data = [[k, list(v)]for k, v in groupby(nusc['annotations'], lambda k: k['image_id'])]

    N = len(sample_data)
    # N = 2000
    # frames = [None] * N
    print('Number of samples:', N)

    limit = []
    N_per_thread = N // num_thread
    for i in range(num_thread):
        limit.append([i*N_per_thread, (i+1)*N_per_thread])
    limit[-1][-1] = N

    # lock = threading.Lock()
    # checker = [0] * N

    list_thread = [
        threading.Thread(
            target=thread_task, 
            args=(
                thread_id, limit[thread_id], cfg, data_split, 
                anchor_prior, nusc, sample_data, 
                # frames,
                save_dir,
                total_objects[thread_id], total_usable_objects[i],
                uniform_sum_each_type[i], uniform_square_each_type[i], 
                preprocess, examine[i], sums[i], squared[i]
            )
        )
        for thread_id in range(num_thread)
    ]

    for th in list_thread:
        th.start()
    
    for th in list_thread:
        th.join()
    
    merged_total_objects = np.array(total_objects).sum(axis=0)
    merged_total_usable_objects = np.array(total_usable_objects).sum(axis=0)
    merged_uniform_sum_each_type = np.array(uniform_sum_each_type).sum(axis=0)
    merged_uniform_square_each_type = np.array(uniform_square_each_type).sum(axis=0)
    merged_examine = np.array(examine).sum(axis=0)
    merged_sums = np.array(sums).sum(axis=0)
    merged_squared = np.array(squared).sum(axis=0)

    
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if data_split == 'training' and anchor_prior:
        
        for j in range(len(cfg.obj_types)):
            global_mean = merged_uniform_sum_each_type[j] / merged_total_objects[j]
            global_var  = np.sqrt(merged_uniform_square_each_type[j] / merged_total_objects[j] - global_mean ** 2)

            avg = merged_sums[j] / (merged_examine[j][:, :, np.newaxis] + 1e-8)
            EX_2 = merged_squared[j] / (merged_examine[j][:, :, np.newaxis] + 1e-8)
            std = np.sqrt(EX_2 - avg ** 2)

            avg[merged_examine[j] < 10, :] = -100  # with such negative mean Z, anchors/losses will filter them out
            std[merged_examine[j] < 10, :] = 1e10
            avg[np.isnan(std)]      = -100
            std[np.isnan(std)]      = 1e10
            avg[std < 1e-3]         = -100
            std[std < 1e-3]         = 1e10

            whl_avg = np.ones([avg.shape[0], avg.shape[1], 3]) * global_mean[3:6]
            whl_std = np.ones([avg.shape[0], avg.shape[1], 3]) * global_var[3:6]

            avg = np.concatenate([avg, whl_avg], axis=2)
            std = np.concatenate([std, whl_std], axis=2)

            npy_file = os.path.join(save_dir,'anchor_mean_{}.npy'.format(cfg.obj_types[j]))
            np.save(npy_file, avg)
            std_file = os.path.join(save_dir,'anchor_std_{}.npy'.format(cfg.obj_types[j]))
            np.save(std_file, std)


    # pkl_file = os.path.join(save_dir,'imdb.pkl')
    # pickle.dump(frames, open(pkl_file, 'wb'))
    print("{} split finished precomputing".format(data_split))



def main(config:str="config/config.py"):
    cfg = cfg_from_file(config)
    torch.cuda.set_device(cfg.trainer.gpu)
    
    read_one_split(cfg, 'validation')
    read_one_split(cfg, 'training')

    print("Preprocessing finished")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
