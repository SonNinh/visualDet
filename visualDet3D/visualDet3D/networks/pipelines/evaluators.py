import os
from torch.nn.modules import module
from tqdm import tqdm
from easydict import EasyDict
from collections import defaultdict
from typing import Sized, Sequence
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from visualDet3D.evaluator.kitti.evaluate import evaluate
from visualDet3D.evaluator.kitti_depth_prediction.evaluate_depth import evaluate_depth
from visualDet3D.networks.utils.utils import BBox3dProjector, BackProjection
from visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt

@PIPELINE_DICT.register_module
@torch.no_grad()
def evaluate_kitti_depth(cfg:EasyDict, 
                       model:nn.Module,
                       dataset_val:Sequence,
                       writer:SummaryWriter,
                       epoch_num:int, 
                       result_path_split='validation'
                       ):
    model.eval()
    result_path = os.path.join(cfg.path.preprocessed_path, result_path_split, 'data')
    if os.path.isdir(result_path):
        os.system("rm -r {}".format(result_path))
        print("clean up the recorder directory of {}".format(result_path))
    os.mkdir(result_path)
    print("rebuild {}".format(result_path))
    for index in tqdm(range(len(dataset_val))):
        data = dataset_val[index]
        collated_data = dataset_val.collate_fn([data])
        image, K = collated_data
        return_dict = model(
                [image.cuda().float(), image.new(K)]
            )
        depth = return_dict["target"][0, 0]
        depth_uint16 = (depth * 256).cpu().numpy().astype(np.uint16)
        w, h = data['original_shape'][1], data['original_shape'][0]
        height_to_pad = h - depth_uint16.shape[0]
        depth_uint16 = np.pad(depth_uint16, [(height_to_pad, 0), (0, 0)], mode='edge')
        depth_uint16 = cv2.resize(depth_uint16, (w, h))
        depth_uint16[depth_uint16 == 0] = 1 
        image_name = "%010d.png" % index
        cv2.imwrite(os.path.join(result_path, image_name), depth_uint16)

    if "is_running_test_set" in cfg and cfg["is_running_test_set"]:
        print("Finish evaluation.")
        return
    result_texts = evaluate_depth(
        label_path = os.path.join(cfg.path.validation_path, 'groundtruth_depth'),
        result_path = result_path
    )
    for index, result_text in enumerate(result_texts):
        if writer is not None:
            writer.add_text("validation result {}".format(index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
        print(result_text, end='')
    print()

@PIPELINE_DICT.register_module
@torch.no_grad()
def evaluate_kitti_obj(cfg:EasyDict, 
                       model:nn.Module,
                       dataset_val:Sized,
                       writer:SummaryWriter,
                       epoch_num:int,
                       result_path_split='validation'
                       ):
    model.eval()
    result_path = os.path.join(cfg.path.preprocessed_path, result_path_split, f'data_{cfg.detector.name}')
    if os.path.isdir(result_path):
        os.system("rm -r {}".format(result_path))
        print("clean up the recorder directory of {}".format(result_path))
    os.mkdir(result_path)
    print("rebuild {}".format(result_path))
    test_func = PIPELINE_DICT[cfg.trainer.test_func]
    projector = BBox3dProjector().cuda()
    backprojector = BackProjection().cuda()
    
    results = defaultdict(list)
    warmup_iter = 100
    for index in tqdm(range(len(dataset_val))):
        is_warming_up = False if index > warmup_iter else True

        sample_token, bboxes_json = test_one(
            cfg, index, dataset_val, model, test_func, 
            backprojector, projector, result_path,
            is_warming_up=is_warming_up
        )
        results[sample_token].extend(bboxes_json)
        
        # if index == 300:
        #     break
    
    print('Pre processing:', model.time_elapsed['pre_process'] / (len(dataset_val)-warmup_iter), '(ms)')
    print('Feed forward:', model.time_elapsed['feed_forward'] / (len(dataset_val)-warmup_iter), '(ms)')
    print('Post processing:', model.time_elapsed['post_process'] / (len(dataset_val)-warmup_iter), '(ms)')
    

    meta = {
        "use_camera":   True,          # Whether this submission uses camera data as an input.
        "use_lidar":    False,       # Whether this submission uses lidar data as an input.
        "use_radar":    False,        # Whether this submission uses radar data as an input.
        "use_map":      False,       # Whether this submission uses map data as an input.
        "use_external": False,      # Whether this submission uses external data as an input.
    }

    submission = {
        'results': results,
        'meta': meta
    }

    jsonString = json.dumps(submission)
    jsonFile = open(os.path.join(result_path, "submission.json"), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print('Finish generating submission file at', result_path)


    if "is_running_test_set" in cfg and cfg["is_running_test_set"]:
        print("Finish evaluation.")
        return

    return
    result_texts = evaluate(
        label_path=os.path.join(cfg.path.data_path, 'label_2'),
        result_path=result_path,
        label_split_file=cfg.data.val_split_file,
        current_classes=[i for i in range(len(cfg.obj_types))],
        gpu=min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    )
    for class_index, result_text in enumerate(result_texts):
        if writer is not None:
            writer.add_text("validation result {}".format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
        print(result_text)

def test_one(cfg, index, dataset, model, test_func, backprojector:BackProjection, projector:BBox3dProjector, result_path, is_warming_up=False):
    data = dataset[index]
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']

    original_height = data['original_shape'][0]
    collated_data = dataset.collate_fn([data])
    height = collated_data[0].shape[2]
        
    scores, bbox, obj_names = test_func(
        collated_data, model, None, cfg=cfg, is_warming_up=is_warming_up
    )
    bbox_2d = bbox[:, 0:4]
    if bbox.shape[1] > 4: # run 3D
        if is_warming_up:
            model.starter.record()

        bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha, bot, top]
        bbox_3d_state_3d = backprojector(bbox_3d_state, P2) #[x, y, z, w,h ,l, alpha, bot, top]

        # theta is Ry in kitti datatset
        _, _, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

        original_P = data['original_P']
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]
        
        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top  = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top

        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y

        if not is_warming_up:
            model.ender.record()
            torch.cuda.synchronize()
            curr_time = model.starter.elapsed_time(model.ender)
            model.time_elapsed['post_process'] += curr_time


        # write_result_to_file(result_path, index, scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)
        bbox_3d_state_3d = bbox_3d_state_3d.tolist()
        scores = scores.tolist()
        thetas = thetas.tolist()

        bboxes_json = []
        for i in range(len(bbox_2d)):
            if bbox_3d_state_3d[i][3] <=0 or bbox_3d_state_3d[i][4] <=0 or bbox_3d_state_3d[i][5] <= 0:
                print(bbox_3d_state_3d[i])
                continue
            
    
            bboxes_json.append({
                'sample_token': data['image_id'],
                'translation': (bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1] + 0.5*bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][2]),
                'size': (bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5]),
                # 'rotation': bbox_3d_state_3d[i][6],
                'rotation': thetas[i],
                'alpha': bbox_3d_state_3d[i][6],
                'velocity': 0.,
                # 'ego_translation': (0., 0., 0.),
                'num_pts': -1,
                'detection_name': obj_names[i],
                'detection_score': scores[i],
                'attribute_name': ''
            })

        return data['image_id'], bboxes_json
    
    else:
        if "crop_top" in cfg.data.augmentation and cfg.data.augmentation.crop_top is not None:
            crop_top = cfg.data.augmentation.crop_top
        elif "crop_top_height" in cfg.data.augmentation and cfg.data.augmentation.crop_top_height is not None:
            if cfg.data.augmentation.crop_top_height >= original_height:
                crop_top = 0
            else:
                crop_top = original_height - cfg.data.augmentation.crop_top_height

        scale_2d = (original_height - crop_top) / height
        bbox_2d[:, 0:4] *= scale_2d
        bbox_2d[:, 1:4:2] += cfg.data.augmentation.crop_top
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        write_result_to_file(result_path, index, scores, bbox_2d, obj_types=obj_names)



'''
backbone res18 dcn
Pre processing: 1.1520177095366395 (ms)
Feed forward: 7.2770444303072574 (ms)
Post processing: 55.90464265773449 (ms)

backbone res18 no_dcn
'''