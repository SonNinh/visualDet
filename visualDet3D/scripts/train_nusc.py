"""
    Script for launching training process
"""
import os
import sys
import numpy as np
from easydict import EasyDict
from torch.serialization import save
from tqdm import tqdm
from fire import Fire
import coloredlogs
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from visualDet3D.evaluator.kitti.evaluate import evaluate
import visualDet3D.data.kitti.dataset
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import LossLogger, cfg_from_file, save_model, load_model
from visualDet3D.networks.optimizers import optimizers, schedulers

def main(config="config/config.py", experiment_name="default", world_size=1, local_rank=-1):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
        experiment_name (str): Custom name for the experitment, only used in tensorboard.
        world_size (int): Number of total subprocesses in distributed training. 
        local_rank: Rank of the process. Should not be manually assigned. 0-N for ranks in distributed training (only process 0 will print info and perform testing). -1 for single training. 
    """

    ## Get config
    cfg = cfg_from_file(config)

    ## Collect distributed(or not) information
    cfg.dist = EasyDict()
    cfg.dist.world_size = world_size
    cfg.dist.local_rank = local_rank
    is_distributed = local_rank >= 0 # local_rank < 0 -> single training
    is_logging     = local_rank <= 0 # only log and test with main process
    is_evaluating  = local_rank <= 0

    ## Setup writer if local_rank > 0
    recorder_dir = os.path.join(
        cfg.path.log_path, 
        experiment_name + f"config={config}"
    )
    if is_logging: # writer exists only if not distributed and local rank is smaller
        ## Clean up the dir if it exists before
        if os.path.isdir(recorder_dir):
            os.system("rm -r {}".format(recorder_dir))
            print("clean up the recorder directory of {}".format(recorder_dir))
        writer = SummaryWriter(recorder_dir)

        ## Record config object using pprint
        import pprint

        formatted_cfg = pprint.pformat(cfg)
        writer.add_text("config.py", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text
    else:
        writer = None

    ## Set up GPU and distribution process
    if is_distributed:
        cfg.trainer.gpu = local_rank # local_rank will overwrite the GPU in configure file
    gpu = min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(cfg.trainer, 'cudnn', False)
    torch.cuda.set_device(gpu)
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(local_rank)
 
    ## define datasets and dataloader.
    dataset_train = DATASET_DICT[cfg.data.train_dataset](cfg)
    dataset_val = DATASET_DICT[cfg.data.val_dataset](cfg, "validation")

    dataloader_train = DataLoader(
        dataset_train, num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size, collate_fn=dataset_train.collate_fn, 
        shuffle=local_rank<0, drop_last=True,
        sampler=torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=world_size, rank=local_rank, shuffle=True
        ) if local_rank >= 0 else None
    )
    dataloader_val = DataLoader(
        dataset_val, num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size, 
        collate_fn=dataset_val.collate_fn, 
        shuffle=False, drop_last=True
    )

    ## Create the model
    detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
    
    ## define optimizer and weight decay
    optimizer = optimizers.build_optimizer(cfg.optimizer, detector)
    start_epoch = 0
    best_loss = sys.maxsize

    ## Load old model if needed
    old_checkpoint = getattr(cfg.path, 'pretrained_checkpoint', None)
    if old_checkpoint is not None:
        # state_dict = torch.load(old_checkpoint, map_location='cpu')
        # detector.load_state_dict(state_dict)
        detector, optimizer, start_epoch, best_loss = load_model(
            detector, old_checkpoint, optimizer, 
            True, cfg.optimizer.lr
        )

    start_epoch += 1

    ## Convert to cuda
    if is_distributed:
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
        detector = torch.nn.parallel.DistributedDataParallel(
            detector.cuda(), device_ids=[gpu], output_device=gpu
        )
    else:
        detector = detector.cuda()
    detector.train()


    ## Record basic information of the model
    if is_logging:
        string1 = detector.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("model structure", string1) # add space for markdown style in tensorboard text
        num_parameters = get_num_parameters(detector)
        print(f'number of trained parameters of the model: {num_parameters}')

    

    ## define scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.trainer.max_epochs, cfg.optimizer.lr_target)
    scheduler_config = getattr(cfg, 'scheduler', None)
    scheduler = schedulers.build_scheduler(scheduler_config, optimizer)
    # is_iter_based = getattr(scheduler_config, "is_iter_based", False)

    ## define loss logger
    training_loss_logger =  LossLogger(writer, 'train') if is_logging else None
    val_loss_logger =  LossLogger(writer, 'test') if is_logging else None
    

    ## training pipeline
    if 'training_func' in cfg.trainer:
        training_dection = PIPELINE_DICT[cfg.trainer.training_func]
    else:
        raise KeyError
    
    if 'val_func' in cfg.trainer:
        val_detection = PIPELINE_DICT[cfg.trainer.val_func]
    else:
        raise KeyError

    ## Get evaluation pipeline
    # if 'evaluate_func' in cfg.trainer:
    #     evaluate_detection = PIPELINE_DICT[cfg.trainer.evaluate_func]
    #     print("Found evaluate function {}".format(cfg.trainer.evaluate_func))
    # else:
    #     evaluate_detection = None
    #     print("Evaluate function not found")


    ## timer is used to estimate eta
    # timer = Timer()

    print('Num training images: {}'.format(len(dataset_train)))

    global_step = 0
    pth_save_best = os.path.join(
        cfg.path.checkpoint_path, 
        '{}_{}_best.pth'.format(experiment_name, cfg.detector.name)
    )
    pth_save_last = os.path.join(
        cfg.path.checkpoint_path, 
        '{}_{}_best.pth'.format(experiment_name, cfg.detector.name)
    )
    

    for epoch_num in range(start_epoch, cfg.trainer.max_epochs):
        ## Start training for one epoch
        detector.train()
        if training_loss_logger:
            training_loss_logger.reset()

        # iter_num = 0
        tqdm_bar = tqdm(dataloader_train)
        for data in tqdm_bar:
            # iter_num += 1
            
            training_dection(
                data, detector, 
                optimizer=optimizer, 
                loss_logger=training_loss_logger, 
                cfg=cfg, is_training=True
            )

            global_step += 1

            # if is_iter_based:
            #     scheduler.step()

            total_loss_avg = training_loss_logger.loss_stats['total_loss'].avg
            # eta = timer.compute_eta(global_step, len(dataloader_train) * cfg.trainer.max_epochs)
            log_str = f'Train at Epoch: {epoch_num} | train loss: {total_loss_avg:.5f}'
            tqdm_bar.set_description(log_str)

            if is_logging and global_step % cfg.trainer.disp_iter == 0:
                ## Log loss, print out and write to tensorboard in main process
                if 'total_loss' not in training_loss_logger.loss_stats:
                    print(f"\nIn epoch {epoch_num}, global_step:{global_step}, total_loss not found in logger.")
                else:
                    # print(log_str, end='\r')
                    writer.add_text("training_log/train", log_str, global_step)
                    training_loss_logger.log(global_step)
            # break

        if cfg.trainer.val_epoch > 0 and epoch_num % cfg.trainer.val_epoch == 0:
            detector.eval()
            if val_loss_logger:
                val_loss_logger.reset()

            
            tqdm_bar = tqdm(dataloader_val)
            with torch.no_grad():
                for data in tqdm_bar:
                    # iter_num += 1
                    val_detection(
                        data, detector,
                        optimizer=None, 
                        loss_logger=val_loss_logger, 
                        cfg=cfg, is_training=False
                    )

                    total_loss_avg = val_loss_logger.loss_stats['total_loss'].avg
                    # eta = timer.compute_eta(iter_num, len(dataloader_val))
                    log_str = f'Val at Epoch: {epoch_num} | val loss: {total_loss_avg:.5f}'
                    # print(log_str, end='\r')
                    tqdm_bar.set_description(log_str)
                    # break

            # total_loss_avg = val_loss_logger.loss_stats['total_loss'].avg
            # log_str = f'Val at Epoch: {epoch_num} | Running loss: {total_loss_avg:1.5f}'
            # print(log_str, end='\r')
            writer.add_text("testing_log/test", log_str, global_step)
            val_loss_logger.log(global_step)


            if is_logging and total_loss_avg < best_loss:
                best_loss = total_loss_avg
                save_model(
                    pth_save_best, 
                    epoch_num,total_loss_avg, 
                    detector, #.module.state_dict() if is_distributed else detector.state_dict(),
                    optimizer=optimizer
                )


        # if not is_iter_based:
        scheduler.step()

        ## save model in main process if needed
        if is_logging:
            save_model(
                pth_save_last,
                epoch_num, total_loss_avg, 
                detector,#.module.state_dict() if is_distributed else detector.state_dict(),
                optimizer=optimizer
            )

        # if is_logging and (epoch_num + 1) % cfg.trainer.save_iter == 0:
        #     torch.save(
        #         detector.module.state_dict() if is_distributed else detector.state_dict(), 
        #         os.path.join(
        #             cfg.path.checkpoint_path, 
        #             '{}_{}.pth'.format(cfg.detector.name,epoch_num)
        #         )
        #     )


            
        ## test model in main process if needed
        # if is_evaluating and evaluate_detection is not None and cfg.trainer.test_iter > 0 and (epoch_num + 1) % cfg.trainer.test_iter == 0:
        #     print("\n/**** start testing after training epoch {} ******/".format(epoch_num))
        #     evaluate_detection(
        #         cfg, 
        #         detector.module if is_distributed else detector, 
        #         dataset_val, 
        #         writer, 
        #         epoch_num
        #     )
        #     print("/**** finish testing after training epoch {} ******/".format(epoch_num))

        if is_distributed:
            torch.distributed.barrier() # wait untill all finish a epoch

        if is_logging:
            writer.flush()

        # break


if __name__ == '__main__':
    Fire(main)
