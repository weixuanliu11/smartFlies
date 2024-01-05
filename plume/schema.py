import datajoint as dj
import hashlib
import numpy as np
from plume.ppo.main import main

# Setting up schema
schema = dj.schema("jq_fly")

@schema
class TrainingConfig(dj.Manual):
    definition = """
    training_config_hash: char(32)
    ---
    algo: varchar(64)
    lr: double
    eps: double
    alpha: double
    gamma: double
    use_gae: bool
    gae_lambda: double
    entropy_coef: double
    value_loss_coef: double
    max_grad_norm: double
    seed: int
    cuda_deterministic: bool
    num_processes: int
    num_steps: int
    ppo_epoch: int
    num_mini_batch: int
    clip_param: double
    log_interval: double
    save_interval: double
    no_cuda: bool
    use_proper_time_limits: bool
    recurrent_policy: bool
    use_linear_lr_decay: bool
    env_name: varchar(64)
    log_dir: varchar(64)
    save_dir: varchar(256)
    dynamic: bool
    eval_type: varchar(64)
    eval_episodes: int
    eval_interval = NULL: int
    weight_decay: double
    rnn_type: varchar(64)
    hidden_size: int
    betadist: bool
    stacking: int
    masking = NULL: varchar(64)
    stride: int
    dataset: blob
    num_env_steps: blob
    qvar: blob
    birthx: blob
    diff_max: blob
    diff_min: blob
    birthx_linear_tc_steps: int
    birthx_max: double
    dryrun: bool
    curriculum: bool
    turnx: double
    movex: double
    auto_movex: bool
    auto_reward: bool
    loc_algo: varchar(64)
    time_algo: varchar(64)
    env_dt: double
    outsuffix: varchar(256)
    walking: bool
    radiusx: double
    diffusion_min: double
    diffusion_max: double
    r_shaping: blob
    wind_rel: bool
    action_feedback: bool
    squash_action: bool
    flipping: bool
    odor_scaling: bool
    stray_max: double
    test_episodes: int
    viz_episodes: int
    model_fname: varchar(256)
    obs_noise: double
    act_noise: double
    cuda: bool
    """
    

    def insert1(self, dict_to_insert):
        # Compute hash
        buffer = ""
        for key in self.heading.secondary_attributes:
            buffer += str(dict_to_insert[key])

        dict_to_insert['training_config_hash'] = hashlib.md5(buffer.encode()).hexdigest()
        super().insert1(dict_to_insert)

@schema
class TrainingResult(dj.Computed):
    definition = """
    -> TrainingConfig
    ---
    
    """

    def make(self, key):
        # key is a dictionary containg the primary key of TrainingConfig [1 row]
        
        # Pull all the secondary attributes from the table into the dictionary
        args = (TrainingConfig & key).fetch1()

        # Pass it on your code
        main(args = args)

        np.random.seed(args['seed'])
        # EXAMPLE COMPLETE TRIANING
        self.insert1(key)


        #
        # if args.betadist:
        #     print("Setting args.squash_action = False")
        #     args.squash_action = False # No squashing when using Beta

        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)
        # ptitle('PPO Seed {}'.format(args.seed))

        # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cudnn.deterministic = True

        # log_dir = os.path.expanduser(args.log_dir)
        # args.eval_log_dir = log_dir + "_eval"
        # utils.cleanup_log_dir(log_dir)
        # utils.cleanup_log_dir(args.eval_log_dir)

        # torch.set_num_threads(1)
        # # gpu_idx = np.random.choice([i for i in range(torch.cuda.device_count())])
        # gpu_idx = 0
        # device = torch.device(f"cuda:{gpu_idx}" if args.cuda else "cpu")

        # # Curriculum hack
        # datasets = args.dataset
        # birthxs = args.birthx
        # qvars = args.qvar
        # diff_maxs = args.diff_max
        # diff_mins = args.diff_min
        # num_env_stepss = args.num_env_steps
        # assert len(datasets) == len(birthxs) 
        # assert len(datasets) == len(qvars) 
        # assert len(datasets) == len(diff_maxs) 
        # assert len(datasets) == len(diff_mins) 
        # assert len(datasets) == len(num_env_stepss) 
        # stage_idx = 0
        # training_log = None
        # eval_log = None
        # args.dataset = datasets[stage_idx] 
        # args.birthx = birthxs[stage_idx] 
        # args.qvar = qvars[stage_idx] 
        # args.diff_max = diff_maxs[stage_idx] 
        # args.diff_min = diff_mins[stage_idx] 
        # args.num_env_steps = num_env_stepss[stage_idx] 


        # envs = make_vec_envs(args.env_name, 
        #                     args.seed, 
        #                     args.num_processes,
        #                     args.gamma, 
        #                     args.log_dir, 
        #                     device, 
        #                     False, 
        #                     args)

        # eval_env = make_vec_envs(
        #     args.env_name,
        #     args.seed + 1000,
        #     num_processes=1,
        #     gamma=args.gamma, 
        #     log_dir=args.log_dir, 
        #     device=device,
        #     args=args,
        #     allow_early_resets=True)

        # actor_critic = Policy(
        #     envs.observation_space.shape,
        #     envs.action_space,
        #     base_kwargs={
        #                 'recurrent': args.recurrent_policy,
        #                 'rnn_type': args.rnn_type,
        #                 'hidden_size': args.hidden_size,
        #                 },
        #     args=args)
        # actor_critic.to(device)

        # agent = PPO(
        #     actor_critic,
        #     args.clip_param,
        #     args.ppo_epoch,
        #     args.num_mini_batch,
        #     args.value_loss_coef,
        #     args.entropy_coef,
        #     lr=args.lr,
        #     eps=args.eps,
        #     max_grad_norm=args.max_grad_norm,
        #     weight_decay=args.weight_decay)

        # # Save args and config info
        # # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-argparse-namespace-as-a-dictionary
        # fname = f"{args.save_dir}/{args.env_name}_{args.outsuffix}_args.json"
        # with open(fname, 'w') as fp:
        #     json.dump(vars(args), fp)


        # # Save model at START of training
        # fname = f'{args.save_dir}/{args.env_name}_{args.outsuffix}.pt.start'
        # torch.save([
        #     actor_critic,
        #     getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        # ], fname)
        # print('Saved', fname)


        # # Curriculum hack
        # num_stages = len(datasets)
        # fname = f'{args.save_dir}/{args.env_name}_{args.outsuffix}.pt'
        # for stage_idx in range(num_stages):
        #     args.dataset = datasets[stage_idx] 
        #     args.birthx = birthxs[stage_idx] 
        #     args.qvar = qvars[stage_idx] 
        #     args.diff_max = diff_maxs[stage_idx] 
        #     args.diff_min = diff_mins[stage_idx] 
        #     args.num_env_steps = num_env_stepss[stage_idx] 
        #     print(f"Stage: {stage_idx}/{num_stages} - {args.dataset} b{args.birthx} q{args.qvar} n{args.num_env_steps}")

        #     if stage_idx > 0: # already made one above
        #         envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
        #                     args.gamma, args.log_dir, device, False, args)
        #     training_log, eval_log = training_loop(agent, envs, args, device, actor_critic, 
        #         training_log=training_log, eval_log=eval_log, eval_env=eval_env)  
            
        #     # Save model after each stage of training
        #     fname = fname.replace('.pt', f'.{args.dataset}.pt')
        #     torch.save([
        #         actor_critic,
        #         # TODO save optimizer weights
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], fname)
        #     print('Saved', fname)
        

        # #### -------------- Done training - now Evaluate -------------- ####
        # if args.eval_type == 'skip':
        #     return

        # actor_critic.to('cpu')
        # args.model_fname = fname

        # # Evaluation
        # # these datasets are not mentioned in the manuscript
        # print("Starting evaluation")
        # datasets = ['switch45x5b5', 
        #             # 'switch15x5b5', 
        #             # 'switch30x5b5', 
        #             'constantx5b5', 
        #             # 'noisy6x5b5', 
        #             'noisy3x5b5']
        # # if args.dataset not in datasets:
        # #     datasets.append(args.dataset)
        # #     datasets.reverse() # Do training data test first
        # args.flipping = False
        # args.dynamic = False
        # args.fixed_eval = True if 'fixed' in args.eval_type else False
        # args.birthx = 1.0
        # args.birthx_max = 1.0 # the fraction of plume data read in during init
        # args.qvar = 0.0 # doesn't matter for fixed
        # args.obs_noise = 0.0
        # args.act_noise = 0.0
        # args.diffusion_max = args.diffusion_min # always test at min diffusion rate
        # args.diffusionx = args.diffusion_max # added on 10/01/23. this is the parameter if called .eval_loop directly. diffusion_min/max is not init'd in training. Default is 1, same as eval
        # for ds in datasets:
        # print(f"Evaluating on dataset: {ds}")
        # args.dataset = ds
        # test_sparsity = True # if 'constantx5b5' in args.dataset else False # always test sparsity
        # test_sparsity = False if 'short' in args.eval_type else test_sparsity
        # evalCli.eval_loop(args, actor_critic, test_sparsity=test_sparsity)
