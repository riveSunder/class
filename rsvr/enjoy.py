import argparse

import gymnasium as gym
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
matplotlib.rcParams["animation.embed_limit"] = 128

import skimage
import skimage.io as sio
import skimage.transform

from rsvr.simple_reservoir import Reservoir, generate_reservoir, calc_fitness

torch.set_default_dtype(torch.float32)


def plot_reservoir_env(reservoir, policy, env, my_cmap=plt.get_cmap("magma"), 
        title="Reservoir control"):

    global subplot_0, subplot_1, subplot_2, subplot_3
    global bounds, fitness

    fig, ax = plt.subplots(1,4, figsize=(9, 4.5), facecolor="white",
        gridspec_kw={"width_ratios": [0.1, 0.3, 0.1, 0.5]})

    my_cmap = plt.get_cmap("magma")

    obs = env.reset()[0]
    obs = torch.tensor(obs, dtype=torch.get_default_dtype())

    activations = [obs*1.0]
    out, res_activations = reservoir(obs, return_activations=True)
    activations.extend(res_activations)
    activations.append(torch.matmul(out, policy))

    len_nodes = len(activations)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    colors = 0 * np.ones(activations[0].shape)
    np_activations = activations[0]
    for layer_count, nodes in enumerate(activations[1:]):
      np_activations = np.concatenate((np_activations, nodes.numpy()))
      colors = np.concatenate((colors, np.ones(nodes.shape)*(1+layer_count)))

    xx = np.arange(colors.shape[0])
    colors = my_cmap(colors/len_nodes)

    low_obs, high_obs = bounds[0], bounds[1]
    low_res, high_res = bounds[2], bounds[3]
    low_act, high_act = bounds[4], bounds[5]

    subplot_0 = ax[0].scatter(xx[:input_size], np_activations[:input_size],
        color=colors[:input_size], alpha=0.25)
    ax[0].set_ylim(low_obs, high_obs)

    subplot_1 = ax[1].scatter(xx[input_size:-output_size], np_activations[input_size:-output_size],
        color=colors[input_size:-output_size], alpha=0.25)
    ax[1].set_ylim(low_res, high_res)

    subplot_2 = ax[2].scatter(xx[-output_size:], np_activations[-output_size:],
        color=colors[-output_size:], alpha=0.5)
    ax[2].set_ylim(low_act, high_act)

    scene = env.render()

    subplot_3 = ax[3].imshow(scene, interpolation="nearest")

    ax[0].set_title("observation\ninputs")
    ax[1].set_title("reservoir\nstates")
    ax[2].set_title("action\noutputs")
    ax[3].set_title("RL env\nscene")

    fig.suptitle(title, fontsize=28)

    for ii in range(4):
        ax[ii].set_xticklabels('')
        if ii == 3:
          ax[ii].set_yticklabels('')

    ax[3].text(16, 81, f"total reward:\n       {fitness:.3f}")
    ax[3].texts[0].set_size(18)
    ax[3].texts[0].set_color("black")
    ax[3].text(15, 80, f"total reward:\n       {fitness:.3f}")
    ax[3].texts[1].set_size(18)
    ax[3].texts[1].set_color("white")
    plt.tight_layout()

    return fig, ax

def update_fig_reservoir_env(i):

    global subplot_0, subplot_1
    global env, reservoir, policy, weights, obs, ax
    global bounds, fitness

    my_cmap = plt.get_cmap("magma")
    #global ax
    action_high = env.action_space.high
    action_low = env.action_space.low


    reservoir_output = reservoir(obs)
    action_raw = torch.matmul(reservoir_output, policy).numpy()

    action = np.clip(action_raw, action_low, action_high)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    obs = torch.tensor(obs, dtype=torch.get_default_dtype())
    
    activations = [obs*1.0]
    out, res_activations = reservoir(obs, return_activations=True)
    activations.extend(res_activations)
    activations.append(torch.matmul(out, policy))
    

    len_nodes = len(activations)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    colors = 0 * np.ones(activations[0].shape)
    np_activations = activations[0]
    for layer_count, nodes in enumerate(activations[1:]):
      np_activations = np.concatenate((np_activations, nodes.numpy()))
      colors = np.concatenate((colors, np.ones(nodes.shape)*(1+layer_count)))

    xx = np.arange(colors.shape[0])
    colors = my_cmap(colors/len_nodes)

    ax[2].cla()
    ax[1].cla()
    ax[0].cla()

    low_obs, high_obs = bounds[0], bounds[1]
    low_res, high_res = bounds[2], bounds[3]
    low_act, high_act = bounds[4], bounds[5]

    subplot_0 = ax[0].scatter(xx[:input_size], np_activations[:input_size],
        color=colors[:input_size], alpha=0.25)
    ax[0].set_ylim(low_obs, high_obs)

    subplot_1 = ax[1].scatter(xx[input_size:-output_size], np_activations[input_size:-output_size],
        color=colors[input_size:-output_size], alpha=0.25)
    ax[1].set_ylim(low_res, high_res)

    subplot_2 = ax[2].scatter(xx[-output_size:], np_activations[-output_size:],
        color=colors[-output_size:], alpha=0.5)
    ax[2].set_ylim(low_act, high_act)

    scene = env.render()

    fitness += reward
    subplot_3.set_array(scene)
    ax[3].texts[-1].remove()
    ax[3].texts[-1].remove()

    ax[3].text(16, 81, f"total reward:\n       {fitness:.3f}")
    ax[3].texts[0].set_size(18)
    ax[3].texts[0].set_color("black")
    ax[3].text(15, 80, f"total reward:\n       {fitness:.3f}")
    ax[3].texts[1].set_size(18)
    ax[3].texts[1].set_color("white")

    ax[0].set_title("observation\ninputs")
    ax[1].set_title("reservoir\nstates")
    ax[2].set_title("action\noutputs")

    for ii in range(4):
        ax[ii].set_xticklabels('')
        if ii == 3:
          ax[ii].set_yticklabels('')

    plt.tight_layout()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("-p", "--policy_filepath", type=str, 
      default="results/invpend_example_1754886563/invpend_example_1754886563_8_31_champion.pt")
  parser.add_argument("-r", "--reservoir_filepath", type=str, 
      default="results/invpend_example_1754886563/invpend_example_1754886563_rsvr.pt",
      help="load policy or population from this filepath")
  parser.add_argument("-x", "--policy_index", type=int, default=0,
      help="visualise performance for policy at population element at `index`")
  parser.add_argument("-d", "--out_dim", type=int, default=64,
      help="number of elements in the reservoir output/policy input")
  parser.add_argument("-n", "--in_dim", type=int, default=4,
      help="input dimensions from the environment observations")
  parser.add_argument("-m", "--hidden_dim", type=int, default=128,
      help="number of elements in the hidden layer")
  parser.add_argument("-e", "--environment_name", type=str,
      default="InvertedPendulum-v5", 
      help="environment for evolving policies. default = InvertedPendulum-v5")
  parser.add_argument("-u", "--human_mode", type=int, default=0,
      help="set to 1 to render in human mode, leave at default 0 to save animation instead")
  parser.add_argument("-b", "--bounds", nargs="+", type=float,
      default=[-0.125, 0.125, -1.0, 1.0, -0.125, 0.125],
      help="y axis limits for obs, res, act (ol, oh, rl, rh, al, ah)")
  parser.add_argument("-f", "--num_frames", type=int, default=1000)

  hid_dim = 128 

  args = parser.parse_args()

  #in_dim = args.in_dim
  out_dim = args.out_dim

  hid_dim = args.hidden_dim
  reservoir_filepath = args.reservoir_filepath
  policy_filepath = args.policy_filepath
  policy_index = args.policy_index
  env_name = args.environment_name
  num_frames = args.num_frames
  bounds = args.bounds

  env = gym.make(env_name, render_mode="rgb_array")

  in_dim = env.observation_space.shape[0]

  reservoir = generate_reservoir(in_dim, out_dim, hid_dim) 
  reservoir.load_state_dict(torch.load(reservoir_filepath))
  policy = torch.load(policy_filepath)[policy_index]

  obs = env.reset()[0]

  policy_name = os.path.splitext(os.path.split(policy_filepath)[-1])[0]
  animation_filepath = os.path.join("results", f"{policy_name}.mp4")

  fitness = 0.
  fig, ax = plot_reservoir_env(reservoir, policy, env)

  matplotlib.animation.FuncAnimation(fig, update_fig_reservoir_env, 
      frames=num_frames, interval=50).save(animation_filepath)

