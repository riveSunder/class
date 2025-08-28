# Cross Labs Alife Summer School

<div align="center">
<img src="docs/assets/curious_solition_s7_config_semistable_frog000_notitle_noname.png" width=50%>
</div>

## CLASS

* [Syllabus](https://rivesunder.github.io/class/syllabus)
* [Slides](https://rivesunder.github.io/class/entree_toc)


## CLASS code

There are 3 Alife-adjacent substrates we've developed for CLASS exploration and experiment. These are

* `SRNCA` - 
* `dw.simple_dw` -
* `rsvr` - 
* and [Golly](https://golly.sourceforge.io/)

## Golly Cellular Automata (CA) Simulator

Golly has a gentle and rewarding learning curve _and_ an expansive coverage of different CA systems and patterns. There is an active community of CA enthusiasts on the [conwaylife.com forums](https://conwaylife.com/forums/). Although John Conway's Game of Life was introduced to the public in 1970 (in [Martin Gardner's "Mathematical Games"](https://www.ibiblio.org/lifepatterns/october1970.html) column in Scientific American, [pdf](https://web.archive.org/web/20250202021933/https://web.stanford.edu/class/sts145/Library/life.pdf)) and has been continuously explored and developed for decades, there are still new artifacts and discoveries being made in Life-like CA. See the forum on [Turing-complete Life-like CA](https://conwaylife.com/forums/viewtopic.php?f=11&t=2597&start=125) to get an idea of the scene. 

Golly is already installed in the computers we've set up for in-person CLASS 2025, but Golly is [freely available](https://sourceforge.net/projects/golly/files/golly/golly-4.3/) if you want to run it locally on your own machine and there is also a [web version](https://golly.sourceforge.io/webapp/golly.html) you can run in your browser. 

Golly supports [Life-like CA](https://en.wikipedia.org/wiki/Life-like_cellular_automaton) (of which there are 262,144 different rule variants), the 256 different rule variants of [elementary CA](https://en.wikipedia.org/wiki/Elementary_cellular_automaton), John von Neumann's [original 29-state automaton](https://en.wikipedia.org/wiki/Von_Neumann_cellular_automaton), as well as more exotic options like loops, Turmites, and non-totalistic CA.

    <div align="center">
      <a href="docs/assets/JvNLoopReplicator.webm">
        <img src="docs/assets/JvNLoopReplicator.jpg" title="JvNLoopReplicator.rle.gz running in Golly. The machine gradually extends 'C-arm' based on instructions flowing in from the long track extending out on the bottom right. Click for a short video of the initial activity of the machine."> 
      </a>
    </div>




## simple daisyworld

`simple_dw` is an implementation of Watson and Lovelock's original 0-dimensional daisyworld model ([WALO1983](), [LO1982]())

## rsvr

```
python -m rsvr.simple_reservoir -h

```

```
usage: simple_reservoir.py [-h] [-e ENVIRONMENT_NAME] [-g GENERATIONS] [-d OUT_DIM] [-i ELITES] [-l LIST_AVAILABLE_ENVS] [-m MUTATION_RATE]
                           [-n NUMBER_RUNS] [-o OUTPUT_FOLDER] [-p POPULATION_SIZE] [-r RANDOM_SEED] [-s RESERVOIR_STEPS] [-t TAG]

options:
  -h, --help            show this help message and exit
  -e ENVIRONMENT_NAME, --environment_name ENVIRONMENT_NAME
                        environment for evolving policies. default = InvertedPendulum-v5
  -g GENERATIONS, --generations GENERATIONS
                        number of generations to run, default: 10
  -d OUT_DIM, --out_dim OUT_DIM
                        number of elements in the reservoir output/policy input
  -i ELITES, --elites ELITES
                        number of elites to keep from each generation, default = 0 (no elitism)
  -l LIST_AVAILABLE_ENVS, --list_available_envs LIST_AVAILABLE_ENVS
                        To list available environments, pass argument '--list_available_envs 1' or '-l 1'
  -m MUTATION_RATE, --mutation_rate MUTATION_RATE
                        mutation rate in avg number of mutations per individual policy each generation. default is 1. This determines the
                        probability of any given element of each policy mutating as `m/out_dim`
  -n NUMBER_RUNS, --number_runs NUMBER_RUNS
                        number of runs per policy to calculate fitness, default: 3
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        folder to store results. default is `results`
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
                        number of individuals in a population at each generation
  -r RANDOM_SEED, --random_seed RANDOM_SEED
                        used to initialise pseudorandom number generator,default = 196884
  -s RESERVOIR_STEPS, --reservoir_steps RESERVOIR_STEPS
                        steps through reservoir, per time step. default is 3
  -t TAG, --tag TAG     tag for labeling experiments default: default_tag
```


```
python rsvr/simple_reservoir.py --list_available_envs 1

```

```
InvertedPendulum-v5
HalfCheetah-v5
```


