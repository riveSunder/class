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


