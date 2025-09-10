import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def validate(global_step, model, val_loader, config, args, logger, writer, scheduler):
    loss_tape = ValidationLossTape()
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            batch = recursive_to(batch, args.device)
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            loss_tape.update(loss_dict, 1)

    avg_loss_easydict = loss_tape.log(global_step, logger, writer, 'val')
    
    # Safely convert to float for the scheduler
    avg_overall_loss_tensor = avg_loss_easydict # It's a tensor, not a dict
    avg_loss_float = avg_overall_loss_tensor.item()
    
    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss_float)
    
    return avg_loss_float

if __name__ == '__main__':
    # --- 1. Argument Parsing & Setup ---
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Number of gradient accumulation steps')
    args = parser.parse_args()
    assert isinstance(args.accumulation_steps, int) and args.accumulation_steps >= 1
    
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # --- 2. Logging, Data, Model, Optimizer, Scheduler Setup (No changes) ---
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args); logger.info(config)
    
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(
        train_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), 
        shuffle=True, num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)


    # --- 3. State Initialization & Resuming ---
    it_first = 1
    global_step_first = 0
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration'] + 1
        global_step_first = ckpt.get('global_step', 0) # Load global_step directly
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        logger.info(f'Resumed at micro-step {it_first}, global_step {global_step_first}.')

    # --- 4. Main Training Loop (RESTRUCTURED FOR EQUIVALENCE) ---
    try:
        # max_iters now clearly means number of parameter updates
        max_global_steps = config.train.max_iters 
        
        # tqdm now tracks global_steps, which is what you care about
        pbar = tqdm(initial=global_step_first, total=max_global_steps, desc='Train', dynamic_ncols=True)
        
        it = it_first
        global_step = global_step_first
        
        optimizer.zero_grad()
        
        # The loop is driven by global_step, achieving your desired number of updates
        while global_step < max_global_steps:
            
            # --- Accumulation Inner Loop ---
            accum_loss_dict_tensor = {}
            for micro_step in range(args.accumulation_steps):
                model.train()
                
                batch = recursive_to(next(train_iterator), args.device)
                
                loss_dict = model(batch)
                raw_loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                
                if not torch.isfinite(raw_loss):
                    # ... (NaN saving logic)
                    raise RuntimeError(f'NaN/Inf loss detected at micro-step {it}')

                # Accumulate detached Tensors
                if 'overall' not in loss_dict: loss_dict['overall'] = raw_loss
                for k, v in loss_dict.items():
                    accum_loss_dict_tensor[k] = accum_loss_dict_tensor.get(k, torch.zeros_like(v)) + v.detach()
                
                (raw_loss / args.accumulation_steps).backward()
                
                it += 1 # Increment micro-step counter

            # --- Parameter Update Step (This block runs ONCE per global_step) ---
            global_step += 1
            pbar.update(1)

            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            
            if config.train.scheduler.type != 'plateau':
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)

            # Log averaged stats for this global_step
            avg_loss_dict_tensor = {k: v / args.accumulation_steps for k, v in accum_loss_dict_tensor.items()}
            
            log_losses(
                avg_loss_dict_tensor, global_step, 'train', logger, writer, 
                others={'grad': orig_grad_norm.item(), 'lr': optimizer.param_groups[0]['lr']}
            )

            # --- Validation and Saving (Driven by global_step) ---
            if global_step > 0 and global_step % config.train.val_freq == 0:
                avg_val_loss = validate(global_step, model, val_loader, config, args, logger, writer, scheduler)
                
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, f'{global_step}.pt')
                    torch.save({
                        'config': config, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'iteration': it - 1,
                        'global_step': global_step, # Save the current global_step
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)

        pbar.close()
        logger.info('Training finished.')
        
    except KeyboardInterrupt:
        logger.info('Terminating...')
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")