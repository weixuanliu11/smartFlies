import tamagotchi.config as config
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

def load_plume(
    dataset='constant', 
    t_val_min=None,
    t_val_max=None,
    env_dt=0.04,
    puff_sparsity=1.00,
    radius_multiplier=1.00,
    diffusion_multiplier=1.00,
    data_dir=config.datadir,
    ):
    print("[load_plume]",dataset)
    puff_filename = f'{data_dir}/puff_data_{dataset}.pickle' 
    wind_filename = f'{data_dir}/wind_data_{dataset}.pickle' 

    # pandas dataframe
    data_puffs = pd.read_pickle(puff_filename)
    data_wind = pd.read_pickle(wind_filename)

    # Load plume/wind data and truncate away upto t_val_min 
    if t_val_min is not None:
        data_wind.query("time >= {}".format(t_val_min), inplace=True)
        data_puffs.query("time >= {}".format(t_val_min), inplace=True)

    # SPEEDUP: **Further** truncate plume/wind data by sim. time
    if t_val_max is not None:
        data_wind.query("time <= {}".format(t_val_max), inplace=True)
        data_puffs.query("time <= {}".format(t_val_max), inplace=True)

    ## Downsample to env_dt!
    env_dt_int = int(env_dt*100)
    assert env_dt_int in [2, 4, 5, 10] # Limit downsampling to these for now!
    if 'tidx' not in data_wind.columns:
        data_wind['tidx'] = (data_wind['time']*100).astype(int)
    if 'tidx' not in data_puffs.columns:
        data_puffs['tidx'] = (data_puffs['time']*100).astype(int)
    data_wind.query("tidx % @env_dt_int == 0", inplace=True)
    data_puffs.query("tidx % @env_dt_int == 0", inplace=True)

    # Sparsify puff data (No change in wind)
    if puff_sparsity < 0.99:
        print(f"[load_plume] Sparsifying puffs to {puff_sparsity}x")
        puff_sparsity = np.clip(puff_sparsity, 0.0, 1.0)
        drop_idxs = data_puffs['puff_number'].unique()
        drop_idxs = pd.Series(drop_idxs).sample(frac=(1.00-puff_sparsity))
        data_puffs.query("puff_number not in @drop_idxs", inplace=True)

    # Multiply radius 
    if radius_multiplier != 1.0:
        print("Applying radius_multiplier", radius_multiplier)
        data_puffs.loc[:,'radius'] *= radius_multiplier

    min_radius = 0.01

    # Adjust diffusion rate
    if diffusion_multiplier != 1.0:
        print("Applying diffusion_multiplier", diffusion_multiplier)
        data_puffs.loc[:,'radius'] -= min_radius # subtract initial radius
        data_puffs.loc[:,'radius'] *= diffusion_multiplier # adjust 
        data_puffs.loc[:,'radius'] += min_radius # add back initial radius

    # Add other columns
    data_puffs['x_minus_radius'] = data_puffs.x - data_puffs.radius
    data_puffs['x_plus_radius'] = data_puffs.x + data_puffs.radius
    data_puffs['y_minus_radius'] = data_puffs.y - data_puffs.radius
    data_puffs['y_plus_radius'] = data_puffs.y + data_puffs.radius
    data_puffs['concentration'] = (min_radius/data_puffs.radius)**3


    return data_puffs, data_wind

def get_concentration_at_tidx(data, tidx, x_val, y_val):
    # find the indices for all puffs that intersect the given x,y,time point
    qx = str(x_val) + ' > x_minus_radius and ' + str(x_val) + ' < x_plus_radius'
    qy = str(y_val) + ' > y_minus_radius and ' + str(y_val) + ' < y_plus_radius'
    q = qx + ' and ' + qy
    d = data[data.tidx==tidx].query(q)
    return d.concentration.sum()

def cleanup_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return
    # try:
    #     os.makedirs(log_dir)
    # except OSError:
    #     files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    #     for f in files:
    #         os.remove(f)
    
def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# New version: One circle at hardcoded (x,y) with quiver 
def plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=False):
    # Instantaneous wind velocity
    # Normalize wind (just care about angle)
    data_at_t = data_wind[data_wind.time==t_val]
    v_x, v_y = data_at_t.wind_x.mean(), data_at_t.wind_y.mean()
    v_xy = np.sqrt(v_x**2 + v_y**2)*20
    v_x, v_y = v_x/v_xy, v_y/v_xy
    # print("v_x, v_y", v_x, v_y)

    # Arrow
    x,y = -0.15, 0.6 # Arrow Center [Note usu. xlim=(-0.5, 8)]
    if invert_colors:
        color='white'
    else:
        color='black'
    ax.quiver(x, y, v_x, v_y, color=color, scale=2.5)
    # ax.quiver(x, y, v_x, v_y, color='black', scale=500)

    # Circle is 1 scatterplot point!
    ax.scatter(x, y, s=500, 
        facecolors='none', 
        edgecolors=color,
        linestyle='--')

def plot_puffs(data, t_val, ax=None, show=True):
    # TODO check color to concentration mapping
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        
    # xmin = -2 #data.x.min()
    # xmax = 12 #data.x.max()
    # ymin = -5 #data.y.min()
    # ymax = +5 #data.y.max()
    # set limits
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    # data_at_t = data[data.time==t_val] # Float equals is dangerous!
    data_at_t = data[np.isclose(data.time, t_val, atol=1e-3)] # Smallest dt=0.01, so this is more than enough!
    # print("data_at_t.shape", data_at_t.shape, t_val, data.time.min(), data.time.max())

    c = data_at_t.concentration
    # print(c, t_val)

    # alphas = (np.log(c+1e-5)+np.abs(np.log(1e-5))).values
    # alphas /= np.max(alphas)
    # alphas = np.clip(alphas, 0.0, 1.0)

    alphas = c.values
    alphas /= np.max(alphas) # 0...1
    alphas = np.power(alphas, 1/8) # See minimal2 notebook
    # alphas = np.power(alphas, 10)
    alphas = np.clip(alphas, 0.2, 0.4)

    alphas *= 2.5/data_at_t.x # decay alpha by distance too
    alphas = np.clip(alphas, 0.05, 0.4)


    rgba_colors = np.zeros((data_at_t.time.shape[0],4))
    # rgba_colors[:,0] = 1.0 # Red
    # rgba_colors[:,2] = 1.0 # Blue
    # https://matplotlib.org/3.1.1/gallery/color/named_colors.html
    # https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    rgba_colors[:,0:3] = matplotlib.colors.to_rgba('gray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkgray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('dimgray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkslategray')[:3] # too dark
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightsteelblue')[:3] # ok
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('red')[:3] 
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightskyblue')[:3] 

    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas

    # fig.canvas.draw()
    # s = ((ax.get_window_extent().width  / (xmax-xmin+1.) * 72./fig.dpi) ** 2)
    k = 6250*((fig.get_figwidth()/8.0)**2) # trial-and-error
    s = k*(data_at_t.radius)**2 
    # print('size', s) # 885

    ax.scatter(data_at_t.x, data_at_t.y, s=s, facecolor=rgba_colors, edgecolor='none')

    if show:
        plt.show()

def plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax=None, fname='', plotsize=(10,10), show=True, invert_colors=False):
    if ax is None:
        fig = plt.figure(figsize=plotsize)
        ax = fig.add_subplot(111)
    plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=invert_colors)
    plot_puffs(data_puffs, t_val, ax, show=False)
    
    if len(fname) > 0:
        # fname = savedir + '/' + 'puff_animation_' + str(idx).zfill(int(np.log10(data['puffs'].shape[1]))+1) + '.jpg'
        fig.savefig(fname, format='jpg', bbox_inches='tight')
        plt.close()
    return fig, ax