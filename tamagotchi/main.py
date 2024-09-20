"""
# Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# take out the TC hack
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import os
import sys
import numpy as np

import argparse
import json
from setproctitle import setproctitle as ptitle

import tamagotchi.data_util as utils
from env import make_vec_envs, get_vec_normalize
from tamagotchi.a2c_ppo_acktr.ppo import PPO
from tamagotchi.a2c_ppo_acktr.model import Policy
from tamagotchi.a2c_ppo_acktr.storage import RolloutStorage
from training import training_loop

def get_args():
    parser = argparse.ArgumentParser(description='PPO for Plume')
    parser.add_argument('--algo', default='ppo')
    parser.add_argument('--lr', type=float, default=7e-4,
        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=137,
        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
        help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
        help='use a linear schedule on the learning rate')
    # My params start
    parser.add_argument('--env-name')
    parser.add_argument('--log-dir', default='/tmp/gym/')
    parser.add_argument('--save-dir', default='./trained_models/')
    parser.add_argument('--dynamic', type=bool, default=False)
    parser.add_argument('--eval_type',  type=str, 
        default=['fixed', 'short', 'skip'][0])
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--rnn_type', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--betadist', type=bool, default=False)
    parser.add_argument('--stacking', type=int, default=0)
    parser.add_argument('--masking', type=str, default=None)
    parser.add_argument('--stride', type=int, default=1)
    # Curriculum variables
    parser.add_argument('--dataset', type=str, nargs='+', default=['constantx5b5'])
    parser.add_argument('--num-env-steps', type=int, default=10e6)
    # parser.add_argument('--num-env-steps', type=int, nargs='+', default=[10e6]) # save for bkwds compat
    parser.add_argument('--qvar', type=float, nargs='+', default=[0.0])
    parser.add_argument('--birthx',  type=float, default=1.0)
    parser.add_argument('--diff_max',  type=float, nargs='+', default=[0.8])
    parser.add_argument('--diff_min',  type=float, nargs='+', default=[0.4])
    parser.add_argument('--birthx_linear_tc_steps', type=int, default=0) # if on, birthx will linearly decrease over time, reachinig the birthx value gradually
    # apparent wind 
    parser.add_argument('--apparent_wind', type=bool, default=False) 
    parser.add_argument('--apparent_wind_allo', type=bool, default=False) 
    parser.add_argument('--visual_feedback', type=bool, default=False) 
    parser.add_argument('--flip_ventral_optic_flow', type=bool, default=False) # PEv3: for eval to see the behavioral impact of flipping course direction perception.
    parser.add_argument('--birthx_max',  type=float, default=1.0) # Only used for sparsity
    parser.add_argument('--dryrun',  type=bool, default=False) 
    parser.add_argument('--curriculum', type=bool, default=False) # not used 
    parser.add_argument('--turnx',  type=float, default=1.0)
    parser.add_argument('--movex',  type=float, default=1.0)
    parser.add_argument('--auto_movex',  type=bool, default=False)
    parser.add_argument('--auto_reward',  type=bool, default=False)
    parser.add_argument('--loc_algo',  type=str, default='uniform')
    parser.add_argument('--time_algo',  type=str, default='uniform')
    parser.add_argument('--env_dt',  type=float, default=0.04)
    parser.add_argument('--outsuffix',  type=str, default='')
    parser.add_argument('--walking',  type=bool, default=False)
    parser.add_argument('--radiusx',  type=float, default=1.0)
    parser.add_argument('--diffusion_min',  type=float, default=1.0)
    parser.add_argument('--diffusion_max',  type=float, default=1.0)
    parser.add_argument('--r_shaping',  type=str, nargs='+', default=['step'])
    parser.add_argument('--wind_rel',  type=bool, default=True)
    parser.add_argument('--action_feedback',  type=bool, default=False)
    parser.add_argument('--squash_action',  type=bool, default=False)
    parser.add_argument('--flipping', type=bool, default=False)
    parser.add_argument('--odor_scaling', type=bool, default=False)
    parser.add_argument('--stray_max', type=float, default=2.0)
    parser.add_argument('--test_episodes',  type=int, default=50)
    parser.add_argument('--viz_episodes',  type=int, default=10)
    parser.add_argument('--model_fname',  type=str, default='')
    parser.add_argument('--obs_noise', type=float, default=0.0)
    parser.add_argument('--act_noise', type=float, default=0.0)
    parser.add_argument('--if_vec_norm', type=int, default=1) # whether to normalize the input
    parser.add_argument('--if_train_actor_std', type=bool, default=False) # whether to train the std of the stochastic policy
    args = parser.parse_args()
    assert len(args.dataset) == len(args.qvar) 
    assert len(args.dataset) == len(args.diff_max) 
    assert len(args.dataset) == len(args.diff_min) 
    # args.cuda = not args.no_cuda and 
    cuda_available = torch.cuda.is_available()
    args.cuda = cuda_available
    print("CUDA:", args.cuda)
    assert args.algo in ['a2c', 'ppo']
    print(args)
    return args

def main(args=None):
    if not args:
        args = get_args()
    else:
        # turn a dict into a parsed arg object.
        args = argparse.Namespace(**args)
        # default_args = get_args()
        # defaults = {param.name: param.default for param in default_args.parameters.values() if param.default != inspect.Parameter.empty}
        # defaults.update({param.name: param.default for param in init_signature_vis_fb_params.parameters.values() if param.default != inspect.Parameter.empty})
        # Write into args if not already present
        # for key, value in defaults.items():
        #     if not hasattr(args, key):
        #         setattr(args, key, value)
        # Only default values
        args.dryrun = False
        args.flip_ventral_optic_flow = False
        
    print("PPO Args --->", args)
    print(args.seed)
        

    np.random.seed(args.seed)
    if args.betadist:
        print("Setting args.squash_action = False")
        args.squash_action = False # No squashing when using Beta

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ptitle('PPO Seed {}'.format(args.seed))

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.", flush=True, file=sys.stderr)
        args.cuda = False
    
    args.model_fname = f"{args.env_name}_{args.outsuffix}.pt"
    args.model_fpath = os.path.join(args.save_dir, 'weights', args.model_fname)
    args.training_log = os.path.join(args.save_dir, 'train_logs', args.model_fname.replace(".pt", '_train.csv'))
    # Save args and config info
    # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-argparse-namespace-as-a-dictionary
    args.json_config = os.path.join(args.save_dir, 'json', args.model_fname.replace(".pt", "_args.json"))
    if not args.dryrun:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'weights'), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'train_logs'), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'chkpt'), exist_ok=True)
            os.makedirs(os.path.join(args.save_dir, 'json'), exist_ok=True)
            
        except OSError:
            raise Exception("Could not create save directory")
        
        with open(args.json_config , 'w') as fp:
            json.dump(vars(args), fp)
            
        log_dir = os.path.expanduser(args.log_dir)
        args.eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(args.eval_log_dir)

    torch.set_num_threads(1)
    gpu_idx = 0
    device = torch.device(f"cuda:{gpu_idx}" if args.cuda else "cpu")

    # Build envs for training
    curriculum_vars = {
        'dataset': args.dataset,
        'qvar': args.qvar,
        'diff_max': args.diff_max,
        'diff_min': args.diff_min,
        'reset_offset_tmax': [30, 3, 30], # 3 for switch condition, according to evalCli 
        't_val_min': [60, 58, 60] # start time of plume data. 58 for switch condition, at around when the switching happens accoding to evalCli
    }
    
    # creates the envs and deploys the first 'num_processes' envs 
    envs = make_vec_envs(
        args.env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        args.log_dir,
        device,
        True,  # allow_early_resets? This is to support resetting an env twice in a row. Twice in a row happens because one auto reset after done and another after swapping.
        args = args,
        **curriculum_vars) # set these envs vars according to the curriculum
    
    if not args.if_vec_norm:
        envs.venv.norm_obs = False

    actor_critic = Policy(
        envs.observation_space.shape, 
        envs.action_space,
        base_kwargs={
                     'recurrent': args.recurrent_policy,
                     'rnn_type': args.rnn_type,
                     'hidden_size': args.hidden_size,
                     },
        args=args)
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)
    

    if not args.dryrun:    
        # Save model at START of training
        start_fname = f'{args.model_fpath}.start'
        torch.save([
            actor_critic,
            getattr(get_vec_normalize(envs), 'ob_rms', None)
        ], start_fname)
        print('Saved', start_fname)
    
    # keeping these for backwards compatibility
    training_log = None
    eval_log = None

    # run training loop
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    training_log, eval_log = training_loop(agent, envs, args, device, actor_critic, 
        training_log=training_log, eval_log=eval_log, eval_env=None, rollouts=rollouts)  

    # close the envs
    try:
        import gc
        gc.collect()
        envs.close()
        del training_log
        del eval_log
    except Exception as e:
        print(f"Cleaning objs after training failed.. see exception message: {e}", flush=True)
        
if __name__ == "__main__":
    main()
