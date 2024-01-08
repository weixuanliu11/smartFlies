import datajoint as dj
import hashlib
import numpy as np
from plume.ppo.main import main
import random

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
    outsuffix: varchar(256)
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
    num_env_steps: int
    qvar: blob
    birthx: double
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
    
    def get_new_seed(self):
        # return a random seed that is NOT already in the table
        while True:
            seed = random.randint(0, 32767) # keep tradition of $RANDOM in bash, where RAND~Uni(0, 32767)
            if seed not in self.fetch()['seed']:
                return seed
    

    def insert1(self, dict_to_insert):
        if 'seed' not in dict_to_insert.keys():
            seed = self.get_new_seed()
            dict_to_insert['seed'] = seed
        
        # dummy outsuffix
        dict_to_insert['outsuffix'] = ''
        
        # Compute hash
        buffer = ""
        for key in self.heading.secondary_attributes:
            buffer += str(dict_to_insert[key])
        dict_to_insert['training_config_hash'] = hashlib.md5(buffer.encode()).hexdigest()
        
        # outsuffix of the form: seed_hash 
        dict_to_insert['outsuffix'] = "_".join([str(seed), dict_to_insert['training_config_hash']])
        
        super().insert1(dict_to_insert)

    def json_insert(self, fjson):
        # TODO insert from json files. In case the table gets wiped from server
        return 
    
    
@schema
class TrainingResult(dj.Computed):
    definition = """
    -> TrainingConfig
    ---
    hours_elapsed: double
    """

    def make(self, key):
        # key is a dictionary containg the primary key of TrainingConfig [1 row]
        
        # start timer
        import time
        t0 = time.time()
        
        # Pull all the secondary attributes from the table into the dictionary
        args = (TrainingConfig & key).fetch1()

        # Pass it on your code
        main(args = args)

        # stop timer
        t1 = time.time()
        # calculate time taken in hours
        t = (t1-t0)/3600

        key['hours_elapsed'] = t
        # EXAMPLE COMPLETE TRIANING
        self.insert1(key)