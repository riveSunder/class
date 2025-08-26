import os
import argparse

import numpy as np
import torch
import scipy


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation

from rsvr.simple_reservoir import Reservoir, calc_spectral_radius

def calculate_damping(transient: torch.tensor):

  fit_fn = lambda x, a,b,c, gamma, delay: (a*np.exp(-b*x)+c)*np.sin(x*gamma-delay)

  results = []
  results_csv = "node_index, lambda, gamma, zeta\n" 

  for node in range(transient.shape[-1]):
    
    try:
      popt, pcov = scipy.optimize.curve_fit(fit_fn, np.arange(transient.shape[0]), transient[:,node], maxfev=120000)
    except:
      print(f"curve fitting failed for node {node}")
      results.append({"node": node, "lambda": "failed", "gamma": "failed", "zeta": "failed"})

    print(f"scipy curve_fit suggests {popt} for neural node {node}")
    zeta_approx = popt[1]/popt[4]
    zeta = popt[1] / np.sqrt(popt[1]**2+popt[4]**2)

    print(f"scipy curve_fit suggests lambda={popt[1]}, freq. gamma={popt[4]}, \n"\
        f"\tdamping ratio ~{zeta_approx:.4e} = {zeta:.4e}")
    results_csv += f"{node}, {popt[1]:.6f}, {popt[4]:.6f}, {zeta:.6f}\n"

    results.append({"node": node, "lambda": popt[1], "gamma": popt[4], "zeta": zeta})

  return results, results_csv

def calculate_transient(x: torch.tensor, weights: torch.tensor, activation: callable,
    max_steps: int=1024) -> torch.tensor:
  """

  """

  transient = activation(torch.matmul(x, weights))
  
  for my_step in range(max_steps-1):
    transient = torch.cat([transient, activation(torch.matmul(transient[-1:,:], weights))])

  return transient

def plot_surf(transient: torch.tensor, 
    sup_title: str="Neural activations transient",
    ax_title: str="") -> tuple:

  fig = plt.figure(figsize=(9,9))
  ax = plt.axes(projection="3d")

  x = np.arange(transient.shape[0])
  y = np.arange(transient.shape[1])

  xx, yy = np.meshgrid(y,x)

  ax.plot_surface(xx, yy, transient, cmap="magma", alpha = 0.9)

  fig.suptitle(sup_title)
  ax.set_title(ax_title)
  return fig, ax

def update_surf(ii):

  global subplot
  global ax

  max_steps = 256
  max_degree_x = 90
  max_degree_y = 10
  
  degree_x = max_degree_x * np.sin(ii * np.pi/max_steps)
  degree_y = max_degree_y * np.cos(ii * np.pi/max_steps)

  ax.view_init(degree_y, degree_x) 

def plot_side(transient: torch.tensor,
    sup_title: str="Neural activations transient",
    ax_title: str="") -> tuple:
  
  fig = plt.figure(figsize=(6,6))
  ax = plt.axes()

  my_cmap = plt.get_cmap("magma")
  number_plots = transient.shape[1]

  for plot_number in range(number_plots): 
    ax.plot(transient[:, plot_number], color=my_cmap(plot_number/number_plots), alpha=0.5)

  fig.suptitle(sup_title)
  ax.set_title(ax_title)
  return fig, ax

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("-r", "--reservoir_filepath", type=str, 
      default="results/inverted_pendulum_CLASS_sd13_rstps3_1756172368/rsvr.pt",
      help="load policy or population from this filepath")
  parser.add_argument("-s", "--transient_steps", type=int,
      default=1024, 
      help="maximum steps for calculating transient")
  parser.add_argument("-t", "--tag", type=str,
      default="default_tag", 
      help="tag for labeling experiments default: default_tag")
  parser.add_argument("-o", "--output_plot", type=str, default=None, 
      help="filename tag to save output figures (surface gif and side plot of transients)"\
          "leave blank/default to not save output figures")
  parser.add_argument("-d", "--damping_tag", type=str, default="default",
      help="use this arg for saving damping info as csv")
  parser.add_argument("-c", "--calc_exp_fit", type=int, default=0)

  args = parser.parse_args()

  reservoir_params = torch.load(args.reservoir_filepath)

  reservoir = Reservoir(**reservoir_params) 
  in_dim = reservoir.weights_in.shape[0]
    
  # 'impulse' input 
  in_x = torch.ones(1,in_dim)
  x = torch.ones(*torch.matmul(in_x, reservoir.weights_in).shape)

  transient = calculate_transient(x, reservoir.weights_hidden.clone(),
      reservoir.act, max_steps=args.transient_steps)

  spectral_radius = calc_spectral_radius(reservoir)

  if args.calc_exp_fit:
    results_a, results_csv = calculate_damping(transient)

    csv_filepath = os.path.join("results", f"damping_{args.damping_tag}.csv")
    with open(csv_filepath, "w") as f:
      f.write(results_csv)
  else:
    ax_title = f"Reservoir transient, "\
        f"spectral radius={spectral_radius:.4f}"
    sup_title = f"{os.path.split(args.reservoir_filepath)[-2]}" 

    fig_side_filepath = os.path.join("results", f"side_{args.output_plot}.png")
    animation_filepath = os.path.join("results", f"surf_{args.output_plot}.mp4")
    gif_filepath = os.path.join("results", f"surf_{args.output_plot}.gif")

    fig_side, ax_side = plot_side(transient, sup_title=sup_title, ax_title=ax_title)
    fig, ax = plot_surf(transient, sup_title=sup_title, ax_title=ax_title)
    
    num_frames = 512
    frame_interval = 33

    fig_side.savefig(fig_side_filepath)
    matplotlib.animation.FuncAnimation(fig, update_surf, 
        frames=num_frames, interval=frame_interval).save(animation_filepath)

    matplotlib.animation.FuncAnimation(fig, update_surf, 
        frames=num_frames, interval=frame_interval).save(gif_filepath)
