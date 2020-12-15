import os
import time
import torch
import random
import numpy as np
import pandas as pd
from utils import args_util, plmodel_util,dataloading
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

import wandb
wandb.init(project="fp_lightning")

@rank_zero_only
def wandb_save(wandb_logger, config):
    wandb_logger.log_hyperparams(config)
    wandb_logger.experiment.save('./pl_fingerprint.py', policy="now")

def main():
    arg_parser = args_util.add_general_args()
    arg_parser = args_util.add_train_args(arg_parser)
    arg_parser = args_util.add_model_args(arg_parser)
    args = arg_parser.parse_args()

    #kishore update parameters
    args_dict = vars(args)
    # args_dict['root_folder'] = r'/home/kishore/Fanyang_code/news/news/outlets'
    args_dict['build_author_predict'] = False
    args_dict['build_topic_predict'] = False

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    config = {'root_folder': args.root_folder,
              'author_dim': args.author_dim,
              'author_track_dim': args.author_track_dim,
              'topic_dim': args.topic_dim,
              'token_dim': args.token_dim,
              'rnn_type': args.rnn_type,
              'rnn_layer': args.rnn_layer,
              'hid_dim': args.hid_dim,
              'dropout': args.dropout,
              'sentiment_dim': args.sentiment_dim,
              'emotion_dim': args.emotion_dim,
              'build_sentiment_embedding': args.build_sentiment_embedding,
              'build_author_emb': args.build_author_emb,
              'build_author_track': args.build_author_track,
              'build_author_predict': args.build_author_predict,
              'build_topic_predict': args.build_topic_predict,
              'leverage_topic': args.leverage_topic,
              'leverage_emotion': args.leverage_emotion,
              'lr': args.lr,
              'epoch': args.epoch,
              'update_iter': args.update_iter,
              'grad_clip': args.grad_clip,
              'use_entire_example_epoch': args.use_entire_example_epoch,
              'batch_size': args.batch_size,
              'update_size': args.update_size,
              'check_step': args.check_step,
              'random_seed': args.random_seed,
              'previous_comment_cnt': args.previous_comment_cnt,
              'min_comment_cnt': args.min_comment_cnt,
              'max_seq_len': args.max_seq_len,
              'max_title_len': args.max_title_len, #kishore_update
              'max_comment_len': args.max_comment_len, #kishore_update
              'prob_to_full': args.prob_to_full,
              'sentiment_fingerprinting': args.sentiment_fingerprinting,
              'emotion_fingerprinting': args.emotion_fingerprinting,
              'freeze_bert' : args.freeze_bert,
              'dataloader_num_workrs': args.dataloader_num_workrs,
              'gpu_id': args.gpu_id
              }
    for key, value in config.items():
        print(key, value)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    #loggers
    wandb_logger = pl.loggers.WandbLogger(project="fp_lightning", log_model=True)
    wandb_save(wandb_logger, config)

    for outlet in [ 'NewYorkTimes'] : #,'Archiveis', 'wsj',]:  # os.listdir(args.root_folder): #kishore_update
        print("Working on {} ...".format(outlet))
       
        #Dataset creation
        input_dataset = dataloading.FinalDataset( output_folder = os.path.join(args.root_folder, outlet),
        MAX_LEN_TITLE = config['max_title_len'], MAX_LEN_COMMENT = config['max_comment_len'])
        #split data to train-test-validation
        input_dataset.datasplit()

        config['author_size'] = len(input_dataset.authors_ar)
        config['topic_size'] = input_dataset.topic_size
        config['outlet'] = outlet

        #create dataloaders
        train_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                    sampler= input_dataset.train_sampler, num_workers= config['dataloader_num_workrs'], pin_memory=True)

        val_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                    sampler= input_dataset.val_sampler, num_workers= config['dataloader_num_workrs'], pin_memory=True)

        test_loader = DataLoader( input_dataset, batch_size= config['batch_size'], \
                    sampler= input_dataset.test_sampler, num_workers= config['dataloader_num_workrs'], pin_memory=True )

        # init model
        fpmodel = plmodel_util.FingerprintModel(config= config)
        wandb_logger.watch( fpmodel, log='gradients', log_freq=100)

        #callbacks
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping( monitor='val_acc', min_delta=0.00, 
        patience=3, verbose=False, mode='max')

        cp_valacc = ModelCheckpoint( filepath=wandb.run.dir+'{epoch:02d}-{val_acc:.2f}', save_top_k=5, monitor='val_acc', mode='max')

        #model summary -- ERROR Exception has occurred: AttributeError 'NotImplementedError' object has no attribute 'message'
        # trainer = pl.Trainer(weights_summary='full')
        # trainer.fit( model = fpmodel, train_dataloader= train_loader, val_dataloaders= val_loader)
        
        #unit test model-- ERROR Exception has occurred: AttributeError 'NotImplementedError' object has no attribute 'message'
        # trainer = pl.Trainer( fast_dev_run = True, gpus = [4] )
        # trainer.fit( model = fpmodel, train_dataloader= train_loader, val_dataloaders= val_loader)

        # run batch size scaling, result overrides hparams.batch_size
        # trainer = pl.Trainer(auto_scale_batch_size = True )
        # # call tune to find the batch size
        # trainer.tune( fpmodel)

        # fast_dev_run = True,
        #debug an epoch run
        # trainer = pl.Trainer( logger = wandb_logger, log_every_n_steps=1, gradient_clip_val = config['grad_clip'], min_epochs = 5, max_epochs = config['epoch'], 
        # val_check_interval = 0.005, callbacks=[early_stop_callback], checkpoint_callback = cp_valacc, auto_scale_batch_size='binsearch', profiler = True, limit_train_batches = 0.7, 
        # accelerator='ddp', plugins='ddp_sharded', gpus = [config['gpu_id']] ,fast_dev_run = True,replace_sampler_ddp=False)  #limit_val_batches=500,

         # Automatically overfit the sane batch of your model for a sanity test
        # trainer = pl.Trainer(overfit_batches=100, logger = wandb_logger,
        #  min_epochs = 5, max_epochs = config['epoch'], 
        #  gpus = [config['gpu_id']] )

        #Model training 
        #using precision = 16, amp_backend='native' -  Exception has occurred: RuntimeError cuDNN error: CUDNN_STATUS_BAD_PARAM
        trainer = pl.Trainer( logger = wandb_logger, log_every_n_steps=1, 
        gradient_clip_val = config['grad_clip'], min_epochs = 5, max_epochs = config['epoch'], 
        val_check_interval = 0.1,limit_val_batches=0.1, 
        callbacks=[early_stop_callback], checkpoint_callback = cp_valacc,
         profiler = True, accelerator='ddp', 
        plugins='ddp_sharded', gpus = [config['gpu_id'], 2] ,replace_sampler_ddp=False
         )  #
        
        # trainer.tune( fpmodel, train_dataloader= train_loader, val_dataloaders= val_loader) # need to add model.batch_size and change dataloader parameter

        # #debugging
        # # use only 10 train batches and 3 val batches
        # trainer = pl.Trainer(limit_train_batches=10, limit_val_batches=3, profiler = "advanced",auto_select_gpus = True, gpus = 3, \
        # accelerator='ddp',plugins='ddp_sharded',replace_sampler_ddp=False )
        # trainer = pl.Trainer(limit_train_batches=10, limit_val_batches=3, profiler = "advanced", gpus = [4], amp_backend = 'native' )

        # # unit test all the code- hits every line of your code once to see if you have bugs, # instead of waiting hours to crash on validation
        # trainer = pl.Trainer(fast_dev_run=True)

        #fitting
        trainer.fit( model = fpmodel, train_dataloader= train_loader, val_dataloaders= val_loader)

        #test
        trainer.test( model= fpmodel, test_dataloaders=test_loader, ckpt_path='best', verbose=True)


if __name__ == '__main__':
    main()