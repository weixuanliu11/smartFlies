import os
import numpy as np
import torch
import gym
from gym.spaces.box import Box
# import gymnasium as gym
# from gymnasium.spaces import Box

from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv as SubprocVecEnv_
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_


import sys, os, importlib
sys.path.append('../')
sys.path.append('../../')


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  args=None,
                  num_frame_stack=None, 
                  **all_curriculum_params
                  ):
    envs = []
    for env_idx in range(len(all_curriculum_params['dataset'])):
        now_curriculum_params = {}
        for k, v in all_curriculum_params.items():
            now_curriculum_params[k] = v[env_idx]
        for i in range(num_processes):
            envs.append(make_env(env_name, seed, i, log_dir, allow_early_resets, args, **now_curriculum_params))

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
        envs.num_envs = num_processes
        envs.deploy(range(num_processes))
    else:
        envs = DummyVecEnv(envs)
        
    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma) # type(envs.action_space) = <class 'gym.spaces.box.Box'>
        
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


def make_env(env_id, seed, rank, log_dir, allow_early_resets, args=None,**kwargs):
    # both seed info is redundant... seed args and provided separately
    if args.dynamic:
        import plume_env_dynamic as plume_env
        # from plume_env_dynamic import PlumeEnvironment, PlumeFrameStackEnvironment
        importlib.reload(plume_env)
        print("Using Dynamic Plume...")
    else:
        # hard coded to be false in evalCli
        import plume_env
        # from plume_env import PlumeEnvironment, PlumeFrameStackEnvironment
        importlib.reload(plume_env)
        print("Using Precomputed Plume...")

    def _thunk():
        if 'plume' in env_id:
            # hard coded to be plume in evalCli: env_name=plume. Only instance of env_id found so far
            if args.recurrent_policy or (args.stacking == 0):
                if kwargs:
                    print("kwargs ON", flush=True, file=sys.stdout)
                    # TODO make this cleaner. auto read kwargs keys and overwrite args to keep args.X format
                    env = plume_env.PlumeEnvironment(
                        dataset=kwargs['dataset'],
                        birthx=kwargs['birthx'],
                        qvar=kwargs['qvar'],
                        diff_max=kwargs['diff_max'],
                        diff_min=kwargs['diff_min'],
                        turnx=args.turnx,
                        movex=args.movex,
                        birthx_max=args.birthx_max,
                        env_dt=args.env_dt,
                        loc_algo=args.loc_algo,
                        time_algo=args.time_algo,
                        auto_movex=args.auto_movex,
                        auto_reward=args.auto_reward,
                        walking=args.walking,
                        radiusx=args.radiusx,
                        r_shaping=args.r_shaping,
                        wind_rel=args.wind_rel,
                        action_feedback=args.action_feedback,
                        squash_action=args.squash_action,
                        flipping=args.flipping,
                        odor_scaling=args.odor_scaling,
                        stray_max=args.stray_max,
                        obs_noise=args.obs_noise,
                        act_noise=args.act_noise,
                        seed=args.seed, 
                        )
                else:
                    # bkw compat before cleaning up TC hack
                    env = plume_env.PlumeEnvironment(
                        dataset=args.dataset,
                        turnx=args.turnx,
                        movex=args.movex,
                        birthx=args.birthx,
                        birthx_max=args.birthx_max,
                        env_dt=args.env_dt,
                        loc_algo=args.loc_algo,
                        time_algo=args.time_algo,
                        diff_max=args.diff_max,
                        diff_min=args.diff_min,
                        auto_movex=args.auto_movex,
                        auto_reward=args.auto_reward,
                        walking=args.walking,
                        radiusx=args.radiusx,
                        r_shaping=args.r_shaping,
                        wind_rel=args.wind_rel,
                        action_feedback=args.action_feedback,
                        squash_action=args.squash_action,
                        flipping=args.flipping,
                        odor_scaling=args.odor_scaling,
                        qvar=args.qvar,
                        stray_max=args.stray_max,
                        obs_noise=args.obs_noise,
                        act_noise=args.act_noise,
                        seed=args.seed, 
                        )
            else:
                # Dont ever see this in logs so far
                print("Using PlumeFrameStackEnvironment...", flush=True, file=sys.stdout)
                env = plume_env.PlumeFrameStackEnvironment(
                    n_stack=args.stacking,
                    masking=args.masking,
                    stride=args.stride if args.stride >= 1 else 'log',
                    dataset=args.dataset,
                    turnx=args.turnx,
                    movex=args.movex,
                    birthx=args.birthx,
                    birthx_max=args.birthx_max,
                    env_dt=args.env_dt,
                    loc_algo=args.loc_algo,
                    time_algo=args.time_algo,
                    diff_max=args.diff_max,
                    diff_min=args.diff_min,
                    auto_movex=args.auto_movex,
                    auto_reward=args.auto_reward,
                    walking=args.walking,
                    radiusx=args.radiusx,
                    r_shaping=args.r_shaping,
                    wind_rel=args.wind_rel,
                    action_feedback=args.action_feedback,
                    squash_action=args.squash_action,
                    flipping=args.flipping,
                    odor_scaling=args.odor_scaling,
                    qvar=args.qvar,
                    stray_max=args.stray_max,
                    obs_noise=args.obs_noise,
                    act_noise=args.act_noise,
                    seed=args.seed,
                    )
        else:
            env = gym.make(env_id)
        env.seed(seed + rank)        

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])
        return env

    return _thunk


from gym import spaces
import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    else:
        return np.stack(obs)


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                # no longer resets if done. Moved to step_async_wait in my SubprocVecEnv
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(SubprocVecEnv_):
    # my version that supports running only a subset of envs 
    # also manages swapping of environments 
    # also manages current MAX delta wind direction
    
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        # inherit __init__ from Gym SubprocVecEnv
        SubprocVecEnv_.__init__(self, env_fns, start_method)
        # lists of remotes and processes that are currently deployed
        self.deployed_remotes = list(self.remotes)
        self.deployed_processes = self.processes
        self.wind_directions = 0 # the MAX number of switches in wind direction
        self.remote_directory = {} # key: index of remote, value: dataset name and deployment status
        available_datasets = self.get_attr_all_envs('dataset') # list of all available datasets 
        
        for i, ds in enumerate(available_datasets):
            self.remote_directory[i] = {'dataset': ds, 'deployed': True}
        
    def sample_wind_dirction(self):
        wind_dir = np.random.randint(0, self.wind_directions)
        return wind_dir
    
    def refresh_deployment_status(self):
        # check all processes and update deployment status
        for i, r in enumerate(self.remotes):
            if r in self.deployed_remotes:
                self.remote_directory[i]['deployed'] = True
            else:
                self.remote_directory[i]['deployed'] = False
        
    def deploy(self, indices: VecEnvIndices = None) -> None:
        """
        Deploy envs in subprocesses.
        :param indices: refers to indices of envs.
        """
        indices = self._get_indices(indices)
        self.deployed_remotes = [self.remotes[i] for i in indices]
        self.deployed_processes = [self.processes[i] for i in indices]
        assert len(indices) == self.num_envs, "cannot deploy more envs than the predetermined num_processes "
        self.refresh_deployment_status()
        
    def ds2_wind(ds):
        # translate dataset name to number of changes in wind direction
        # input: dataset name
        # output: number of changes in wind direction (3 as in the agent may encounter up to 3 different wind directions)
        if 'noisy' in ds:
            return 3
        elif 'constant' in ds:
            return 1
        elif 'switch' in ds:
            return 2
        else:
            raise NotImplementedError
        
    def swap(self, idx: int, idx_replacement_item: int) -> None:
        """
        Swap envs in subprocesses.
        :param indices: refers to indices of envs.
        """
        self.deployed_remotes[idx] = self.remotes[idx_replacement_item]
        self.deployed_processes[idx] = self.processes[idx_replacement_item]
        
    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.deployed_remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.deployed_remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        for i, d in enumerate(dones): 
            if d:
                # log the final observation
                infos[i]["terminal_observation"] = obs[i]
                # decide if swap
                new_wind_direction = self.sample_wind_dirction()
                current_wind_direction = self.get_attr('dataset', i)
                # print(self.get_attr('dataset'))
                self.swap(i, 3)
                # print(self.get_attr('dataset'))

                list(obs)[i] = self.reset_deployed_at(i)
                obs = tuple(obs)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        for idx, remote in enumerate(self.deployed_remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.deployed_remotes]

    def reset(self) -> VecEnvObs:
        # only reset deployed envs
        # TODO is this problematic?
        for remote in self.deployed_remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.deployed_remotes]
        return _flatten_obs(obs, self.observation_space)
    
    def reset_deployed_at(self, indices: VecEnvIndices = None) -> VecEnvObs:
        # reset only envs at indices
        indices = self._get_indices(indices)
        for i in indices:
            self.deployed_remotes[i].send(("reset", None))
        obs = [self.deployed_remotes[i].recv() for i in indices]
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        # unchanged. Not sure when it is called automatically. Never called manually
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        # unchanged. Not sure when it is called automatically. Never called manually
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from the DEPLOYED environments."""
        target_remotes = self._get_target_remotes(indices, deployed=True)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]
    
    def get_attr_all_envs(self, attr_name: str) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = len(self.remotes)
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices, deployed=True)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()
            
    def set_attr_all_env(self, attr_name: str, value: Any) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        indices = len(self.remotes)
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices, deployed: bool = False) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        if deployed:
            return [self.deployed_remotes[i] for i in indices]
        return [self.remotes[i] for i in indices]


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# # Can be used to test recurrent policies for Reacher-v2
# class MaskGoal(gym.ObservationWrapper):
#     def observation(self, observation):
#         if self.env._elapsed_steps > 0:
#             observation[-2:] = 0
#         return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True
    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs
    def train(self):
        self.training = True
    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
