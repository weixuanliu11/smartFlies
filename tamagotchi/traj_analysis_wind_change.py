#%%
# the line above allows debug in interactive mode - can show matplotlib plot when debugging
from __future__ import division
import os
import glob
from natsort import natsorted
import os
import sys
import numpy as np
import tqdm
import pandas as pd
import numpy as np
import glob
import sys
sys.path.append('/src/tamagotchi')
import tamagotchi.config as config
import importlib
import tamagotchi.eval.log_analysis as log_analysis
importlib.reload(log_analysis)
batchmode = False
import argparse
# Common
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")


def get_eval_dfs_and_stack_them(model_fname, use_datasets, number_of_eps, exp_dir='eval', verbose=False):
  is_recurrent = True
  # load eval episodes from pkl files
  model_dir = model_fname.replace('.pt', '/').replace("weights", exp_dir) 
  if not os.path.exists(model_dir):
      print(f"Model directory {model_dir} does not exist")
      sys.exit(0)
  # read pkl file into PD dataframe - each row is an epoch
  selected_df = log_analysis.get_selected_df(model_dir, 
                                [use_datasets], 
                                n_episodes_home=number_of_eps, 
                                n_episodes_other=number_of_eps,
                                balanced=False,
                              #   oob_only=False,
                                min_ep_steps=0)
  if verbose:
    print("model_dir", model_dir)
    logfiles = natsorted(glob.glob(model_dir + '*.pkl'))
    model_seed = model_dir.rstrip('/').split('/')[-1].split('_')[1]
    print("model_seed ---->", model_seed)
    print(f"Found {len(logfiles)} .pkl evaluation logs in {model_dir}")
    print(f"selected N eps {selected_df.shape}")
    print(f"Episode breakdown: \n {selected_df.groupby(['dataset', 'outcome']).count()}")

  # get traj data and stack them
  traj_dfs = []
  squash_action = True
  # for episode_log in tqdm.tqdm(selected_df['log']):
  for idx,row in tqdm.tqdm(selected_df.iterrows()):
      dataset = row['dataset']
      episode_log = row['log']
      # use get_traj_df_tmp to calculate head direction
      traj_df = log_analysis.get_traj_df_tmp(episode_log, 
                                        extended_metadata=True, 
                                        squash_action=squash_action)
      traj_df['idx'] = np.arange(traj_df.shape[0], dtype=int)
      traj_df['ep_idx'] = row['idx']
      traj_df['dataset'] = dataset
      traj_df['outcome'] = row['outcome']
      traj_df['loc_x_dt'] = traj_df['loc_x'].diff() 
      traj_df['loc_y_dt'] = traj_df['loc_y'].diff() 
      traj_dfs.append(traj_df)

  traj_df_stacked = pd.concat(traj_dfs, ignore_index=True)
  if verbose:
    print(f"Stacked traj dfs shape: {traj_df_stacked.shape}")
  return traj_df_stacked


def calc_time_since_last_wind_change(eps_df):
    """
    Calculates the time since the last change in wind direction.

    Parameters:
    - eps_df (pandas.DataFrame): The input DataFrame of eval trajectories for a single episode, containing extended metadata.s

    Returns:
    - pandas.DataFrame: The input DataFrame with an additional column 'time_since_last_wind_change'
                        representing the time since the last change in wind direction.

    Raises:
    - AssertionError: If the time interval between consecutive 't_val' values is not constant.

    """
    # sanity check - delta on t_val should be the same 
    t_val_dt = eps_df['t_val'].diff().dropna().unique()
    t_val_dt = set(np.round(t_val_dt, 2))
    assert len(t_val_dt) == 1, f"t_val_dt is not constant: {t_val_dt}"
    t_val_dt = t_val_dt.pop()
    # first row is always nan - fill with 0
    eps_df['wind_angle_ground_theta_dt1'].iloc[0] = 0
    # calculate time since last change in wind direction 
    time_since_last_wind_change = []
    time_count = 0
    for idx, row in eps_df.iterrows():
        time_count += t_val_dt
        if row['wind_angle_ground_theta_dt1']: # delta theta is not 0 - wind just changed
            time_count = 0    
        time_since_last_wind_change.append(time_count)
    
    eps_df = eps_df.assign(time_since_last_wind_change = time_since_last_wind_change)
    return eps_df


# mark when the wind regime changes
def get_wind_change_regimes(traj_df, wind_change_frame_threshold=5, frame_rate=0.04, verbose=False):
    threshold = wind_change_frame_threshold * frame_rate
    traj_df['wind_regime'] = 'tracking'
    # wind just changed within the last N frames - anemotactic behavior should follow 
    traj_df['wind_regime'].loc[ traj_df['time_since_last_wind_change'] <= threshold ] = 'anemotactic'
    if verbose:
        print("Annotation of wind regimes:")
        print(f"{traj_df['wind_regime'].value_counts()}")
        print(f"Threshold for wind change regime is {threshold} seconds \n")
    return traj_df['wind_regime']


def plot_action_distributions_by_wind_odor_regime(traj_df_stacked, dataset, save_path=False, verbose=False):
    def add_wind_odor_regime(traj_df_stacked):
        traj_df_stacked['odor_wind_regime'] = 'NA'
        traj_df_stacked['odor_wind_regime'][(traj_df_stacked['wind_regime']=='anemotactic') & (traj_df_stacked['regime']=='TRACK')] = 'anemotactic, on plume'
        traj_df_stacked['odor_wind_regime'][(traj_df_stacked['wind_regime']=='anemotactic') & (traj_df_stacked['regime']!='TRACK')] = 'anemotactic, off plume'
        traj_df_stacked['odor_wind_regime'][(traj_df_stacked['wind_regime']=='tracking') & (traj_df_stacked['regime']=='TRACK')] = 'tracking, on plumes'
        traj_df_stacked['odor_wind_regime'][(traj_df_stacked['wind_regime']=='tracking') & (traj_df_stacked['regime']!='TRACK')] = 'tracking, off plume'
        # order rows by wind odor regime name
        wind_regimes = traj_df_stacked['odor_wind_regime'].unique().tolist()
        wind_regimes.sort()
        wind_regimes[1], wind_regimes[2] = wind_regimes[2], wind_regimes[1] # reorder to move ON plume together 
        return wind_regimes # a list of possible wind regimes 
    
    # read in global plotting parameters
    global analysis_columns
    global titles_dict 
    global ticks_dict
    global ticklabels_dict 

    wind_regimes = add_wind_odor_regime(traj_df_stacked) # add wind odor regime column ansd return list of possible regimes
    if verbose:
        print(f"breakdown of wind regimes: \n {traj_df_stacked['odor_wind_regime'].value_counts()} \n")
    n_cols = len(analysis_columns)
    n_regimes = len(wind_regimes)
    # plot action distributions by wind odor regime
    fig, axes = plt.subplots(n_regimes, n_cols, sharex=False, sharey=False, figsize=(9,2.65))
    gs = matplotlib.gridspec.GridSpec(nrows=n_regimes, 
                        ncols=n_cols, 
                        figure=fig, 
                        width_ratios=[1]*n_cols,
                        height_ratios=[1]*n_regimes,
                        )

    traj_df_stacked_subset = traj_df_stacked.query(f"dataset == @dataset")
    E=len(traj_df_stacked_subset['ep_idx'].unique())

    for i, col in enumerate(analysis_columns):
        for j, regime in enumerate(wind_regimes): 
            ax=axes[j, i]
            traj_df_stacked_subset = traj_df_stacked.query("odor_wind_regime == @regime and dataset == @dataset")
            sns.kdeplot(traj_df_stacked_subset[col],
                        ax=ax)
            ax.set_yticks([])
            [ax.spines[s].set_visible(False) for s in ax.spines]
            ax.set_ylabel('')        
            N=len(traj_df_stacked_subset)
            suffix = f"\nN={N}"
            # suffix = f"\nN={N}\nE={E}"
            if i == 0:
                prefix = "$\mathbf{{{}}}$".format(regime.split(', ')[0].lower().capitalize())
                # print(prefix)
                ax.set_ylabel(r"{}".format(prefix) + "{}".format("\n" + regime.split(', ')[1]) + suffix, fontsize=10, 
                                rotation=0, ha='center', labelpad=30, va='center')
            if j < len(wind_regimes)-1:
                ax.set_xticks([])
                ax.set_xlabel('')
            else:
                ax.set_xticks(ticks_dict[col])
                ax.set_xlabel(titles_dict[col], fontsize=10,)
                if col in ticklabels_dict.keys():
                    ax.set_xticklabels(ticklabels_dict[col])
                    if col == 'agent_angle_ground_theta': # same HD calc as sat
                        ax.set_xticklabels(ticklabels_dict[col], rotation=90)
                    elif col == 'ego_course_direction_theta':
                        ax.set_xticklabels(ticklabels_dict[col], rotation=90)
    fig.subplots_adjust(hspace=.1, wspace=0.25)
    if save_path:
        trial_types = "_".join(traj_df_stacked_subset['outcome'].unique())
        save_path = save_path.replace('DUMMY', trial_types)
        print("Saving:", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def load_centerline_df(traj_df_stacked, dataset='noisy3x5b5', verbose=False):
    traj_df_stacked['tidx'] = (traj_df_stacked['t_val']*100).astype(int)
    centerline_filename = f'{config.datadir}/centerline_data_{dataset}.pickle' # centerline angle in [0,1] see smartFlies/plume/centerline_cli.py
    centerline_df = pd.read_pickle(centerline_filename)
    env_dt_int = 4
    centerline_df.query("tidx % @env_dt_int == 0", inplace=True)
    tidx_min = traj_df_stacked['tidx'].min()
    tidx_max = traj_df_stacked['tidx'].max()
    if verbose: print(f"total centerline df {centerline_df.shape}")
    centerline_df.query(" @tidx_min <= tidx <= @tidx_max ", inplace=True)
    if verbose: 
        print(f"subset over time centerline df {centerline_df.shape}")
        print(f"breakdown of centerline angles: \n{centerline_df['angle'].value_counts()}")
    return centerline_df


# Exclude timesteps when wind is blowing close to 0-degrees in the switch once case
def trim_small_wind_angles(df, degrees_omit=15):
    def deg_to_01(deg):
        assert deg >= -180 and deg <= 180
        return 0.5 + deg/360
    wag_exclude_min = deg_to_01(-degrees_omit)
    wag_exclude_max = deg_to_01(+degrees_omit)
    print(f"Trimming out {wag_exclude_min} to {wag_exclude_max}")
    df_return = df.query("(wind_angle_ground_theta <= @wag_exclude_min) or (wind_angle_ground_theta >= @wag_exclude_max)")
    return df_return


def filter_traj_by_wind_regime_and_add_centerline_angle(traj_df_stacked, centerline_df, wind_regime, dataset='noisy3x5b5', verbose=False):
    # based on subset_centerline_angle in /src/notebooks/report_regime_dists.ipynb
    # init column
    traj_df_stacked['centerline_angle'] = 0
    if verbose: print(f"traj df before filter by {wind_regime}: {traj_df_stacked.shape}")
    traj_df_stacked_subset = traj_df_stacked.query("wind_regime == @wind_regime and dataset in @dataset")[:]
    # not used keep in case needed later 
    # if dataset in ['switch45x5b5']:
    #     traj_df_stacked_subset = trim_small_wind_angles(traj_df_stacked_subset, degrees_omit=15)
    #     # see trimmed wind angles
    #     # traj_df_stacked['wind_angle_ground_theta'].hist(bins=50)
    #     # traj_df_stacked['wind_angle_ground_theta'].describe()
    if verbose: print(f"traj df after filter by {wind_regime}: {traj_df_stacked_subset.shape}")
    for idx, row in tqdm.tqdm(traj_df_stacked_subset.iterrows()):
        loc_x = row['loc_x']
        tidx = row['tidx']
        s = centerline_df.query('tidx == @tidx and @loc_x-0.02 <= x <= @loc_x+0.02')['angle'].median()
        traj_df_stacked_subset.loc[idx, 'centerline_angle'] = s
    return traj_df_stacked_subset



def roll_around_angle(a):
    if a <= 1.0 and a >= 0.0:
        return a
    elif a > 1.0:
        return a - 1.0
    else:
        return 1.0 + a


def augment_agent_angle_centerline(traj_df_stacked_subset, verbose=False):
    traj_df_stacked_subset['agent_angle_centerline'] = traj_df_stacked_subset['agent_angle_ground_theta'] - (traj_df_stacked_subset['centerline_angle'] - 0.5)    
    traj_df_stacked_subset['agent_angle_centerline'] = traj_df_stacked_subset['agent_angle_centerline'].apply(roll_around_angle)
    if verbose:
        print("agent_angle_centerline (HD wrt centerline direction) statistics:")
        print(traj_df_stacked_subset['agent_angle_centerline'].describe())
    

def augment_agent_angle_wind(traj_df_stacked_subset, verbose=False):
    traj_df_stacked_subset['agent_angle_wind'] = traj_df_stacked_subset['agent_angle_ground_theta'] - (traj_df_stacked_subset['wind_angle_ground_theta'] - 0.5)    
    traj_df_stacked_subset['agent_angle_wind'] = traj_df_stacked_subset['agent_angle_wind'].apply(roll_around_angle)
    traj_df_stacked_subset['agent_angle_wind'].describe()
    if verbose:
        print("agent_angle_wind (HD wrt wind direction) statistics:")
        print(traj_df_stacked_subset['agent_angle_centerline'].describe())

def plot_head_direction_wrt_wind_and_plume(traj_df_stacked_subset, regime_label, dataset, save_path):
    
    # setup plotting parameters
    label_dict = {
        'agent_angle_ground_theta': 'HD (Ground)',
        'agent_angle_centerline': f'HD (Plume\ncenterline)',
        'agent_angle_wind': 'HD (Wind)',        
    }
    title_dataset_dict = {
        'constantx5b5': 'constant',
        'noisy3x5b5': 'switch-many',
        'switch45x5b5': 'switch-once',
    }
#     fig = plt.figure(figsize=(4,3)) 
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    
    cols_show = [
#             'agent_angle_ground',
             'agent_angle_centerline',
             'agent_angle_wind', 
                ]
    cols_colors = ['b', 'orange', 'g']
    
    # plot 
    for idx, col in enumerate(cols_show):
        sns.distplot(traj_df_stacked_subset[col],
                     hist=False,
                     rug=False,
                     ax=ax,
                     label=label_dict[col],
                     hist_kws=dict(alpha=0.9))
        
    # vertical lines on MODEs
    for idx, kdeline in enumerate(ax.lines):
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        mode_idx = np.argmax(ys)
        ax.vlines(xs[mode_idx], 0, ys[mode_idx], color=cols_colors[idx], ls=':', lw=2)

    plt.xlim(0,1)
    ax = plt.gca()
    ax.set_xticks(ticks_dict['agent_angle_ground_theta'])
    ax.set_xlabel('Head ' + r'direction [$^{\circ}$]', fontsize=14,)
    ax.set_xticklabels(ticklabels_dict['agent_angle_ground_theta'])

    # Add vertical lines
    plt.axvline(x=0.5, c='grey', ls=':', label=r'$\pm 180^{\circ}$', lw=3)

    ax.set_yticks([])
#     ax.set_ylabel('')

    if 'noisy' in dataset:
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=1.0)

    N=len(traj_df_stacked_subset)
    E=len(traj_df_stacked_subset['ep_idx'].unique())
    title = f"{title_dataset_dict[dataset]}, {regime_label} (N={N},E={E})"
    
    plt.title(title)
    
    if save_path:
        print("Saving:", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Course Direction Angle - add CD wrt centerline and wind
def augment_course_angle_centerline(traj_df_stacked_subset, verbose=False):
    traj_df_stacked_subset['course_angle_centerline'] = traj_df_stacked_subset['ego_course_direction_theta'] - (traj_df_stacked_subset['centerline_angle'] - 0.5)    
    traj_df_stacked_subset['course_angle_centerline'] = traj_df_stacked_subset['course_angle_centerline'].apply(roll_around_angle)
    traj_df_stacked_subset['course_angle_centerline'].describe()
    if verbose:
        print("course_angle_centerline (CD wrt plume centerline direction) statistics:")
        print(traj_df_stacked_subset['course_angle_centerline'].describe())

def augment_course_angle_wind(traj_df_stacked_subset, verbose=False):
    traj_df_stacked_subset['course_angle_wind'] = traj_df_stacked_subset['ego_course_direction_theta'] - (traj_df_stacked_subset['wind_angle_ground_theta'] - 0.5)    
    traj_df_stacked_subset['course_angle_wind'] = traj_df_stacked_subset['course_angle_wind'].apply(roll_around_angle)
    if verbose:
        print("course_angle_wind (CD wrt wind direction) statistics:")
        print(traj_df_stacked_subset['course_angle_centerline'].describe())

# Plot Course angle distribution wrt centerline and wind
def plot_course_direction_wrt_wind_and_plume(traj_df_stacked_subset, regime_label, dataset, save_path):
    global label_dict
    global ls_dict
    title_dataset_dict = {
        'constantx5b5': 'constant',
        'noisy3x5b5': 'switch-many',
        'switch45x5b5': 'switch-once',
    }
    label_dict = {
        'ego_course_direction_theta': 'CD (Ground)',
        'course_angle_centerline': f'CD (Plume\ncenterline)',
        'course_angle_wind': 'CD (Wind)',        
    }
    ls_dict = {
        'ego_course_direction_theta': ':',
        'course_angle_centerline': '-',
        'course_angle_wind': '--',        
    }
#     fig = plt.figure(figsize=(4,3)) 
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    
    cols_show = [
#         'course_angle_ground',
             'course_angle_centerline',
             'course_angle_wind', 
                ]
    cols_colors = ['b', 'orange', 'g']

    for idx, col in enumerate(cols_show):
        sns.distplot(traj_df_stacked_subset[col],
                     hist=False,
                     rug=False,
                     ax=ax,
                     label=label_dict[col],
                     kde_kws={'ls':ls_dict[col], 'lw':2.5},
                     hist_kws=dict(alpha=0.9)) 


    plt.xlim(0,1)
    ax = plt.gca()
    ax.set_xticks(ticks_dict['ego_course_direction_theta'])
    ax.set_xlabel('Course ' + r'direction [$^{\circ}$]', fontsize=14,)
    ax.set_xticklabels(ticklabels_dict['ego_course_direction_theta'])
    # Add vertical lines
#     plt.axvline(x=0.5, c='grey', ls=':', label=r'$\pm 180^{\circ}$', lw=2.5)
    plt.axvline(x=0.5, c='grey', ls=':', lw=2.5)
    ax.set_yticks([])
    if 'noisy' in dataset:
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=1.0)

    N=len(traj_df_stacked_subset)
    E=len(traj_df_stacked_subset['ep_idx'].unique())
    title = f"{title_dataset_dict[dataset]}, {regime_label} (N={N},E={E})"
    plt.title(title)
    if save_path:
        print("Saving:", fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    

def arg_parse():
    parser = argparse.ArgumentParser(description='Plot head direction and course direction distributions')
    parser.add_argument('--model_fname', type=str, default='/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_ALL_noisy_wind_0.001/weights/plume_951_23354e57874d619687478a539a360146.pt', help='Path to the model .pt file. For eval traj retrival and plot storage')
    parser.add_argument('--dataset', type=str, default='noisy3x5b5', help='Dataset to use')
    parser.add_argument('--number_of_eps', type=int, default=80, help='Number of episodes to use')
    parser.add_argument('--verbose', type=bool, default=False, help='Print verbose output')
    parser.add_argument('--save', type=bool, default=False, help='Save graphs or just a dry run')
    parser.add_argument('--eval_folder', type=str, default='eval', help='The directory name of the evaluation experiment, where th pkl files are stored in')
    parser.add_argument('--out_dir', type=str, help='folder name of where plots should be saved to')
    parser.add_argument('--wind_change_regime_threshold', type=int, default=10, help='Threshold for wind change regime in seconds')
    parser.add_argument('--regimes', default=['anemotactic', 'tracking'], help='Wind regimes to plot')

    args = parser.parse_args()
    args.model_seed = args.model_fname.rstrip('/').split('/')[-1].split('_')[1]
    args.model_dir = args.model_fname.replace('.pt', '/').replace("weights", args.eval_folder) # output everything to perturb_along_all
    args.out_dir = f"{args.model_dir}/{args.out_dir}" # typically RUN_NAME/eval/report_regime_dists
    # set up output directory and check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Model directory {args.model_dir} does not exist")
        sys.exit(0)
    print("model_dir", args.model_dir)
    print("out_dir", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    return args


def load_plotting_parameters():        
    ''' 
    function for setting up plotting parameters
    '''

    mpl.rcParams['figure.dpi'] = 100
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 18}
    matplotlib.rc('font', **font)


    # https://seaborn.pydata.org/generated/seaborn.set_color_codes.html#seaborn.set_color_codes
    sns.color_palette()
    sns.set_color_codes(palette='deep')

    '''
    plotting setup
    '''

    ## Set plotting parameters
    analysis_columns = [
    'agent_angle_ground_theta',
    'loc_x_dt',
    'loc_y_dt',
    'ego_course_direction_theta',
    'turn',
    'step',
    'stray_distance',
    ]
    titles_dict = {'agent_angle_ground_theta':'Head\n' + r'direction [$^{\circ}$]',
                'loc_x_dt':r'$\Delta$x [m]',
                'loc_y_dt':r'$\Delta$y [m]',
                'ego_course_direction_theta':'Course\n' + r'direction [$^{\circ}$]',
                'turn':'Turn\nAction [R-L] ',
                'step':'Move\nAction [0-1]',
                'stray_distance':'Stray\nDistance [m]'
                } # use as a 
    ticks_dict = {'agent_angle_ground_theta':[0, 0.25, 0.5, 0.75, 1.0],
                'loc_x_dt':[-0.1, 0, 0.1],
                'loc_y_dt':[-0.1, 0, 0.1],
                'ego_course_direction_theta':[0, 0.25, 0.5, 0.75, 1.0],
                'stray_distance':[0, 1, 2],
                'turn':[0,1],
                'step':[0,1],
                }
    ticklabels_dict = {
    #     'agent_angle_ground':[0, r'$\frac{-\pi}{2}$', r'$\pm\pi$', r'$\frac{+\pi}{2}$', 0],
        'agent_angle_ground_theta':[0, r'$-90$', r'$\pm$180', r'$+90$', 0],
        'ego_course_direction_theta':[0, r'$-90$', r'$\pm$180', r'$+90$', 0],
        'turn':['R','L'],
                }
    xlims_dict = {
        'agent_angle_ground_theta':(0.,1.),
    #     'stray_distance':(0.,2.),
    }
        
    return analysis_columns, titles_dict, ticks_dict, ticklabels_dict, xlims_dict

if __name__ == '__main__':
    args = arg_parse()
    # plot the distritbutions of actions taken by the agent wrt wind and odor regimes
    analysis_columns, titles_dict, ticks_dict, ticklabels_dict, xlims_dict = load_plotting_parameters()
    # get traj data and stack them
    traj_df_stacked = get_eval_dfs_and_stack_them(args.model_fname, args.dataset, args.number_of_eps, exp_dir=args.eval_folder, verbose=True) 
    # for each episode, calculate the time since the last wind change
    traj_df_stacked = traj_df_stacked.groupby(traj_df_stacked['ep_idx']).apply(calc_time_since_last_wind_change).reset_index(drop=True)
    traj_df_stacked['wind_regime'] = get_wind_change_regimes(traj_df_stacked, wind_change_frame_threshold=args.wind_change_regime_threshold, frame_rate=0.04, verbose=True)
    fname = False
    if args.save:
        fname = f"{args.out_dir}/action_dist_wind_odor_regimes_DUMMY_{args.model_seed}.png" # "DUMMY" will be replaced by the trial types
        plot_action_distributions_by_wind_odor_regime(traj_df_stacked, args.dataset, save_path=fname, verbose=True)
    # plot head direction and course direction wrt wind and plume centerline
    # load the centerline data
    centerline_df = load_centerline_df(traj_df_stacked, dataset=args.dataset, verbose=False)
    # filter by wind regime and plot HD wrt wind and plume
    for regime in args.regimes:
        traj_df_stacked_subset_by_wind_regime = filter_traj_by_wind_regime_and_add_centerline_angle(traj_df_stacked, centerline_df, regime, dataset='noisy3x5b5', verbose=True)
        # TODO generate this info when saving df - this takes a while. do not want to repeat everytime
        augment_agent_angle_centerline(traj_df_stacked_subset_by_wind_regime, verbose=True) # add this info to the traj_df
        augment_agent_angle_wind(traj_df_stacked_subset_by_wind_regime, verbose=True)       # add this info to the traj_df
        # same for CD
        augment_course_angle_centerline(traj_df_stacked_subset_by_wind_regime, verbose=True) # add this info to the traj_df
        augment_course_angle_wind(traj_df_stacked_subset_by_wind_regime, verbose=True) # add this info to the traj_df
        for odor_wind_regime in traj_df_stacked_subset_by_wind_regime['odor_wind_regime'].unique():
            traj_df_stacked_subset_by_wind_regime_subset_by_odor = traj_df_stacked_subset_by_wind_regime.query("odor_wind_regime == @odor_wind_regime")
            if args.save:
                fname_regime_name = odor_wind_regime.replace(', ', '_')
                fname = f"{args.out_dir}/HD_dist_{args.model_seed}_{args.dataset}_{fname_regime_name}.png"
            plot_head_direction_wrt_wind_and_plume(traj_df_stacked_subset_by_wind_regime_subset_by_odor, odor_wind_regime, args.dataset, save_path=fname)
            if args.save:
                fname = f"{args.out_dir}/CD_dist_{args.model_seed}_{args.dataset}_{fname_regime_name}.png"
            plot_course_direction_wrt_wind_and_plume(traj_df_stacked_subset_by_wind_regime_subset_by_odor, odor_wind_regime, args.dataset, save_path=fname)