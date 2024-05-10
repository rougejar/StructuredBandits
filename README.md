# Structured Bandit Simulator

A custom, incomplete implementation of UCB, UCB-C, and UCB-C-Entropy. The latter two are derived from the structured bandit problem described [here](https://arxiv.org/pdf/1810.08164.pdf). The original source code is [here](https://github.com/shreyasc-13/structured_bandits).
## Requirements

Make sure you have Python installed on your PC. You will then need to create a virtual environment in the project directory (titled UCB-C).

The requirements for the environment are listed in the <tt>requirements.txt</tt> file and can be installed by entering a terminal in the main directory and typing:
```
pip install -r requirements.txt
```

## Instructions

Once you have the environment set up, you can then run the <tt>main.py</tt> script.
To simply run the program, enter a terminal in the main directory and type:
```
python main.py
```
This will run the UCB, UCB-C, and UCB-C-Entropy code for 1000 iterations on a random meta-user. However, you can input extra arguments to modify this behavior. You can see descriptions by typing:
```
python main.py -h
```
It will output:
```
usage: main.py [-h] [-r R] [-algo ALGO] [-age AGE] [-occ OCC]

Structured Bandit Simulator

options:
  -h, --help  show this help message and exit
  -r R        Number of rounds to run the algorithm(s). Default is 1000.
  -algo ALGO  The algorithm(s) to run. Type u for UCB, c for UCB-C, i for Informative-UCB-C. Can run multiple with multiple inputs. Default enables all.
  -age AGE    Age of meta user to test UCB-C on (if enabled). Default is random.
  -occ OCC    Occupation of meta user to test UCB-C on (if enabled). Default is random.
```

The commands in brief are stated above, but again listed are:

If you want to set the number of rounds, type `-r <rounds>`
To run a specific algorithm, type `-algo <algorithm>`
To run UCB-C or UCB-C-Entropy on a specific meta-user, type an age and occupation using `-age <age>` and `-occ <occupation>`
Check <tt>Datasets/README</tt> to see the possible options.

Example:
```
python main.py -r 2000 -algo uc -age 1 -occ 1
```
This will run the code for 2000 iterations, on algorithms UCB and UCB-C on meta-user (1,1), which is the age-occupation combo: (Under 18, "academic/educator")

Specific Examples:
To obtain the graphs shown in the reports, use these commands:
```
python main.py -r 5000 -algo uc -age 35 -occ 4
```

```
python main.py -r 1000 -algo uci -age 1 -occ 1
```

```
python main.py -r 1000 -algo ci -age 50 -occ 16
```