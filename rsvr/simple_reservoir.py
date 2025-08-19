import argparse

import os
import time
import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

class Reservoir(nn.Module):

  def __init__(self,
      weights_in: torch.tensor,
      weights_out: torch.tensor,
      weights_hidden: torch.tensor,
      reservoir_steps: int=1, echo_only: bool=False):
    super().__init__()

    self.reservoir_steps = torch.nn.Parameter(
        torch.tensor(reservoir_steps), requires_grad=False)

    self.echo_only = torch.nn.Parameter(
        torch.tensor(echo_only, dtype=torch.bool), requires_grad=False)

    self.weights_in = torch.nn.Parameter(weights_in, requires_grad=False)
    self.weights_out = torch.nn.Parameter(weights_out, requires_grad=False)
    self.weights_hidden = torch.nn.Parameter(weights_hidden, requires_grad=False)
    self.act = torch.tanh

  def forward(self, obs: np.ndarray, return_activations: bool=False):

    if type(obs) is not torch.Tensor:
      obs = torch.tensor(obs, dtype=torch.get_default_dtype())

    inputs = torch.matmul(obs, self.weights_in)
    hidden = self.act(inputs)

    if return_activations:
      activations = [1.0 * hidden]


    for reservoir_step in range(self.reservoir_steps):

      if self.echo_only:
        hidden = torch.matmul(hidden, self.weights_hidden)
      else:
        hidden = inputs + torch.matmul(hidden, self.weights_hidden)

      hidden = self.act(hidden)
      if return_activations:
        activations.append(1.0 * hidden)

    out = torch.matmul(hidden, self.weights_out)
    out = self.act(out)
    if return_activations:
      activations.append(1.0 * hidden)

    if return_activations:
      return out, activations
    else:
      return out

def generate_reservoir(in_dim: int, out_dim: int, hid_dim: int,
      reservoir_steps: int=1) -> nn.Module:

  # he initializaiton standard deviation

  he_in = torch.sqrt(torch.tensor((6 / (in_dim + hid_dim))))
  he_hidden = torch.sqrt(torch.tensor(6 / (2*hid_dim)))
  he_out = torch.sqrt(torch.tensor(6 / (hid_dim + out_dim)))

  weights_in = torch.rand(in_dim, hid_dim) 
  weights_in *= 2 * he_in 
  weights_in -= he_in

  weights_out = torch.rand(hid_dim, out_dim) 
  weights_out *= 2*he_out
  weights_out -= he_out

  weights_hidden = torch.rand(hid_dim, hid_dim)
  weights_hidden *= 2 * he_hidden
  weights_hidden -= he_hidden

  reservoir = Reservoir(weights_in, weights_out, weights_hidden,
      reservoir_steps=reservoir_steps, echo_only=False)

  return reservoir


def generate_population(population_size: int, reservoir_out_dim: int,
    action_dim: int, connection_probability: float=0.5) -> list:
    
  population_probs = torch.rand(population_size, reservoir_out_dim, action_dim)
  population = 1.0 * (population_probs > connection_probability)

  return population

def calc_fitness(reservoir: nn.Module, policy: torch.Tensor,
      env: gym.wrappers.common.TimeLimit, number_runs: int=3, 
      seed_factor: int=131) -> float:

  action_high = env.action_space.high
  action_low = env.action_space.low

  fitness = 0.0

  for run in range(number_runs):
    obs, info = env.reset(seed=run*seed_factor)
    done = 0
    
    while not done:
      reservoir_out = reservoir(obs)
      action_raw = torch.matmul(reservoir_out, policy).numpy()
      action = np.clip(action_raw, action_low, action_high)
      obs, reward, terminated, truncated, info = env.step(action)

      done = terminated or truncated

      fitness += reward

  return fitness / number_runs

def mutate(population: list, mutation_rate: float, elites: int):
  # mutate

  mutations = 1.0 * (torch.rand(population.shape) < mutation_rate)
  new_population = (population - mutations).abs()
  population[elites:] = new_population[elites:]

  return population

def select_fitness_proportional(fitness_list: list, population: list, 
      mutation_rate: float, elites: int=0) -> list:

  fitness_tensor = torch.tensor(fitness_list, dtype=torch.get_default_dtype())
  sorted_indices = torch.argsort(fitness_tensor, descending=True)
  fitness_tensor -= fitness_tensor.min()
  ps = len(population)

  probs = (fitness_tensor/fitness_tensor.max()).numpy() #torch.softmax(fitness_tensor, -1).numpy()

  probs = probs.astype(np.float64)
  probs /= probs.sum()

  new_ids = np.random.choice(np.arange(len(population)),
      p=probs,
      size=ps)

  for elite_index in range(elites):
    if elite_index >= ps:
      print("Warning: number of elites {elites} is greater than population size {ps}."
          "/n/t Population composed entirely of elites.")
      break

    new_ids[elite_index] = sorted_indices[elite_index]

  new_population = population[new_ids]
  
  new_population = mutate(new_population, mutation_rate, elites)

  return new_population
  
def evolve(population_size: int, number_generations: int, my_seed: int,
      out_dim: int, exp_tag: str, env_name: str, results_path: str,
      reservoir_steps: int, number_runs: int, mutation_rate: float, elites: int):

  # initialise PRNG
  torch.manual_seed(my_seed)
  # just in case
  np.random.seed(my_seed)

  checkpoint_every = max([1, number_generations // 8])

  env = gym.make(env_name)
  action_dim = env.action_space.shape[0]
  in_dim = env.observation_space.shape[0]
  hid_dim = 128 

  reservoir = generate_reservoir(in_dim, out_dim, hid_dim, reservoir_steps)
  population = generate_population(pop_size, out_dim, action_dim)

  # where to save results
  results_folder = os.path.join(results_path, exp_tag)
  if os.path.exists(results_folder):
    pass
  else:
    os.mkdir(results_folder)

  filepath_reservoir = os.path.join(results_folder, f"rsvr.pt")
  torch.save(reservoir.state_dict(), filepath_reservoir)

  for gen in range(number_generations):
  
    fitness_list = [calc_fitness(reservoir, mem, env, number_runs=number_runs,
        seed_factor=gen+my_seed) for mem in population]

    msg = f"max/min mean+/-std. dev. for generation {gen}: "
    msg += f"{np.max(fitness_list):.3f}/{np.min(fitness_list):.3f}, "
    msg += f"{np.mean(fitness_list):.3f}+/-{np.std(fitness_list):.3f}"

    print(msg)
    population = select_fitness_proportional(fitness_list, population, 
        mutation_rate, elites)

  # save results

    if gen % checkpoint_every == 0 or gen == (generations-1):

      print(f"...checkpoint...")
      #checkpoint number
      ckpt = gen // checkpoint_every

      if gen == (generations-1):
        ckpt += 1

      filepath_population = os.path.join(results_folder, f"gen{gen}_ckpt{ckpt}_pop.pt")
      filepath_champion = os.path.join(results_folder, f"gen{gen}_ckpt{ckpt}_champ.pt")

      fitness_list = [calc_fitness(reservoir, mem, env, number_runs=number_runs,
          seed_factor=gen+my_seed) for mem in population]

      fitness_tensor = torch.tensor(fitness_list, requires_grad=False)
      sorted_indices = torch.argsort(fitness_tensor, descending=True)
      population = population[sorted_indices]

      torch.save(population, filepath_population)
      torch.save(population[0:1], filepath_champion)

      print(f"checkpoint saved at"
          f"\n\t {filepath_population}"
          f"\n\t {filepath_champion}")

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("-e", "--environment_name", type=str,
      default="InvertedPendulum-v5", 
      help="environment for evolving policies. default = InvertedPendulum-v5")
  parser.add_argument("-g", "--generations", type=int, default=10,
      help="number of generations to run, default: 10")
  parser.add_argument("-d", "--out_dim", type=int, default=64,
      help="number of elements in the reservoir output/policy input")
  parser.add_argument("-i", "--elites", type=int, default=0,
      help="number of elites to keep from each generation, default = 0 (no elitism)")
  parser.add_argument("-l", "--list_available_envs", type=int, default=0,
      help="To list available environments, " 
          "pass argument '--list_available_envs 1' or '-l 1'")
  parser.add_argument("-m", "--mutation_rate", type=int, default=1,
      help="mutation rate in avg number of mutations per individual policy each "
          "generation. default is 1. This determines the probability of any "
          "given element of each policy mutating as `m/out_dim`")
  parser.add_argument("-n", "--number_runs", type=int, default=3,
      help="number of runs per policy to calculate fitness, default: 3" )
  parser.add_argument("-o", "--output_folder", type=str, default="results",
      help="folder to store results. default is `results`")
  parser.add_argument("-p", "--population_size", type=int, default=32,
      help="number of individuals in a population at each generation")
  parser.add_argument("-r", "--random_seed", type=int, default=196884,
      help="used to initialise pseudorandom number generator," 
          "default = 196884")
  parser.add_argument("-s", "--reservoir_steps", type=int, default=3,
      help="steps through reservoir, per time step. default is 3")
  parser.add_argument("-t", "--tag", type=str,
      default="default_tag", 
      help="tag for labeling experiments default: default_tag")
  
  args = parser.parse_args()
  my_seed = args.random_seed
  pop_size = args.population_size
  env_name = args.environment_name
  out_dim = args.out_dim
  number_runs = args.number_runs
  generations = args.generations
  mutation_rate = args.mutation_rate / out_dim
  elites = args.elites
  tag = args.tag
  results_path = args.output_folder
  reservoir_steps = args.reservoir_steps

  time_tag = str(int(time.time()))

  exp_tag = f"{tag}_sd{my_seed}_rstps{reservoir_steps}_{time_tag}"

  if args.list_available_envs:
    print("list_available_envs == 1, " 
        "will list envs and then quit")

    for elem in gym.registry:
      print(elem)

    print("list_available_envs == 1, " 
        "envs listed above, quitting now")
  else:

    evolve(population_size=pop_size, number_generations=generations,
        my_seed=my_seed, out_dim=out_dim, exp_tag=exp_tag, env_name=env_name, 
        results_path=results_path, reservoir_steps=reservoir_steps, number_runs=number_runs,
        mutation_rate=mutation_rate, elites=elites)

