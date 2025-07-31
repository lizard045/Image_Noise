import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.amp import autocast
import matplotlib.pyplot as plt

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

# GPU優化導入
from optimization_config import get_optimal_config, apply_performance_profile
import torch.backends.cudnn as cudnn


'''
# --------------------------------------------
# training code for DRUNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
'''


def main(json_path='options/train_drunet.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--performance_profile', type=str, default='balanced', 
                       choices=['ultra_fast', 'balanced', 'highq_uality', 'memory_efficient'],
                       help='Performance profile for optimization')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    
    # ----------------------------------------
    # GPU優化設定
    # ----------------------------------------
    if torch.cuda.is_available():
        # 啟用cuDNN基準測試模式
        cudnn.benchmark = True
        cudnn.deterministic = False
        
        # 設定GPU記憶體分配策略
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # 獲取最佳配置
        gpu_config = get_optimal_config()
        gpu_config = apply_performance_profile(gpu_config, parser.parse_args().performance_profile)
        
        print(f" GPU優化啟用:")
        print(f"   批次大小建議: {gpu_config['batch_size']}")
        print(f"   工作線程數: {gpu_config['num_workers']}")
        print(f"   混合精度: {gpu_config['use_amp']}")
        print(f"   JIT編譯: {gpu_config['enable_jit_compile']}")
        
        # 動態調整配置
        if 'dataloader_batch_size' in opt['datasets']['train']:
            suggested_batch = gpu_config['batch_size']
            current_batch = opt['datasets']['train']['dataloader_batch_size']
            if suggested_batch != current_batch:
                print(f"   建議調整批次大小: {current_batch} → {suggested_batch}")
                opt['datasets']['train']['dataloader_batch_size'] = suggested_batch
                
        if 'dataloader_num_workers' in opt['datasets']['train']:
            suggested_workers = gpu_config['num_workers']
            current_workers = opt['datasets']['train']['dataloader_num_workers']
            if suggested_workers != current_workers:
                print(f"   建議調整工作線程: {current_workers} → {suggested_workers}")
                opt['datasets']['train']['dataloader_num_workers'] = suggested_workers

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                # GPU優化的DataLoader設定
                pin_memory = torch.cuda.is_available()
                prefetch_factor = gpu_config.get('prefetch_factor', 2) if torch.cuda.is_available() else 2
                
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=pin_memory,
                                          prefetch_factor=prefetch_factor,
                                          persistent_workers=True if dataset_opt['dataloader_num_workers'] > 0 else False)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    
    # GPU優化：混合精度訓練
    use_amp = torch.cuda.is_available()
    
    # JIT編譯優化（如果支援）
    if torch.cuda.is_available() and gpu_config.get('enable_jit_compile', False) if 'gpu_config' in locals() else False:
        try:
            # 嘗試JIT編譯模型
            dummy_input = torch.randn(1, 4, 128, 128).cuda()
            model.netG = torch.jit.trace(model.netG, dummy_input)
            print(" JIT編譯成功啟用")
        except Exception as e:
            print(f"  JIT編譯失敗，使用標準模式: {e}")
    
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())
        if use_amp:
            logger.info("🚀 混合精度訓練已啟用")
        
        # 顯示GPU記憶體使用情況
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU記憶體使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    best_psnr = -1
    best_step = 0
    no_improve_count = 0
    early_stop_patience = 10  # 連續幾次未提升就 early stop
    psnr_list = []
    loss_list = []
    start_time = time.time()

    for epoch in range(1000000):  # keep running
        if opt['rank'] == 0:
            print(f"\n===== [Epoch {epoch}] Start =====")
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            # 使用 AMP 加速訓練，但讓模型自己處理最佳化
            with autocast('cuda', enabled=use_amp):
                model.optimize_parameters(current_step)
            
            # GPU記憶體管理
            if current_step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                # 顯示 loss
                if 'G_loss' in logs:
                    loss_list.append(logs['G_loss'])
                    message += f" | G_loss: {logs['G_loss']:.4f}"
                # 顯示目前最佳 PSNR
                if len(psnr_list) > 0:
                    message += f" | Best PSNR: {max(psnr_list):.2f}"
                # 預估剩餘時間和GPU利用率
                elapsed = time.time() - start_time
                steps_per_print = opt['train']['checkpoint_print']
                est_total_steps = 50000  # 可根據你預期的總 step 調整
                est_total_time = elapsed / (current_step/steps_per_print) * (est_total_steps/steps_per_print) if current_step > 0 else 0
                est_left = est_total_time - elapsed
                message += f" | Elapsed: {elapsed/60:.1f}min, Est. Left: {est_left/60:.1f}min"
                
                # GPU狀態監控
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                    message += f" | GPU: {gpu_utilization:.1f}% ({gpu_memory_used:.2f}GB/{gpu_memory_total:.1f}GB)"
                
                print(message)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                psnr_list.append(avg_psnr)
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
                # 儲存最佳模型
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_step = current_step
                    logger.info(f'\n[Best Model Updated] step {current_step}, PSNR={avg_psnr:.2f}dB\n')
                    print(f'\n[Best Model Updated] step {current_step}, PSNR={avg_psnr:.2f}dB\n')
                    
                    # 儲存到標準位置
                    model.save('best')
                    
                    # 同時複製到 model_zoo 資料夾
                    import shutil
                    best_model_path = os.path.join(opt['path']['models'], 'best_G.pth')
                    model_zoo_path = os.path.join('model_zoo', f'my_drunet_color_step{current_step}_psnr{avg_psnr:.2f}.pth')
                    if os.path.exists(best_model_path):
                        os.makedirs('model_zoo', exist_ok=True)
                        shutil.copy2(best_model_path, model_zoo_path)
                        logger.info(f'Best model copied to: {model_zoo_path}')
                        print(f'Best model copied to: {model_zoo_path}')
                    
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                # Early stopping
                if no_improve_count >= early_stop_patience:
                    logger.info(f'\n[Early Stopping] step {current_step}, best PSNR={best_psnr:.2f}dB at step {best_step}\n')
                    print(f'\n[Early Stopping] step {current_step}, best PSNR={best_psnr:.2f}dB at step {best_step}\n')
                    # 儲存 loss/psnr 曲線
                    plt.figure()
                    plt.plot(loss_list, label='Loss')
                    plt.xlabel('Print Step')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig('loss_curve.png')
                    plt.close()
                    plt.figure()
                    plt.plot(psnr_list, label='PSNR')
                    plt.xlabel('Test Step')
                    plt.ylabel('PSNR')
                    plt.legend()
                    plt.savefig('psnr_curve.png')
                    plt.close()
                    exit()

if __name__ == '__main__':
    main()
