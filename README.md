# purdue-internship
Code produced during my internship at Purdue. 
The repository contains:

- A program `main.py` which can be used for Markov sources simulations,
  LZ78 compression and quantitative visualizations of these processes.

- Several libraries written in Python that are used by `main.py` : 
    - `eigenvalues.py`
    - `experiments.py`
    - `lempelziv.py`
    - `markov.py`
    - `neininger.py`
    - `normal.py`
    - `szpan.py`

- Another program, `tails.py` coupled with `parallel_tails.py`
  and `experimenttails.py`, which
  allows to simulate a different model for Markov sources (the Markov
  Independent model), and make experiments regarding tail symbols.

- Jupyter notebooks which were used to compute some expressions
  or experiment with the subject
    - `computing_lambda.ipynb`
    - `datastructure_experiment.ipynb`
    - `markov_julia.ipynb`
    - `parallel_tails.ipynb`
    - `pi_computation.ipynb`
    - `tail_symbols.ipynb`
    - `variance_expression.ipynb`

- Two LaTeX source repositories without the style file needed
  for compilation, though they both contain a compiled version :
    - `rapport_jacquet` is a report on my numerical
      simulations
    - `flexible_paper` is a report on a paper doing an analysis of
      the Flexible Parsing algorithm


## Requirements 
Python3 and `pip`. A list of python modules is given 
in `requirements.txt`. Some experiments
in Julia might be encountered too.

## Install

- Recommended, using [pipenv](https://docs.pipenv.org/)
    
        pipenv shell
        pip install -r requirements.txt

- else, using pip
    
        pip install -r requirements.txt --user

## Usage

### Markov source simulation

These two commands will generate a random Markov chain of size 2,
generate words from the corresponding Markov source, and save those
in a file `<filesave>`. These simulations will be used by the next
batch of commands. You can skip this and try some of the datafiles
already in the `datas` folder.

- `python main.py -s <filesave>`

    _Runs a simulation after prompting for three coupless (n_word, n_exp)
    and saves it as `<filesave>`._

- `python main.py -range <filesave>`

    _Runs a simulation after prompting for a range of values ns
    and saves it as `<filesave>`._


### Visualization

Plotting graphs and histograms using the previously generated data. 

- `python main.py <datafile>`

    _Plots the usual histogram_chains from the set of experiments in `<datafile>`._

- `python main.py -m --file <datafile>`

    _Loads the set of experiments `<datafile>` and plots the graphs related
    to mean analysis._

- `python main.py -m --save <savefile>`

    _Runs the analysis and then saves the data - with analysis - into
    the file `<savefile>`._

- `python main.py -v  (--file <datafile> | --save <savefile>)`

    _Works the same as the -m argument previously seen, except that it
    does variance analysis and plots._

- `python main.py -cdf  (--file <datafile> | --save <savefile>)`

    _Works the same as the -m argument previously seen, except that it
    does cumulative distribution function analysis and plots._

## Examples

Generate data and save it in `datas/new_exp.npy`

        python main.py -s datas/new_exp.npy

Analysis of `exp1.npy`

        python main.py exp1.npy

Variance analysis of `exp2.npy`

        python main.py -v --file exp2.npy
