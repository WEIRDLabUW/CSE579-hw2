# Homework 2

## Setup and Installation

    conda env create -f environment.yml
    conda activate cse579a1


## Running the assignment

    python main.py --task pg/actor_critic/sac --env pendulum/ant

(Aliases: `pg = policy_gradient`, `ac = actor_critic`.)

Examples:

    python main.py --task pg --env pendulum
    python main.py --task actor_critic --env pendulum
    python main.py --task sac --env pendulum

Append `--test` to evaluate a saved checkpoint instead of training.

## Files you need to touch:
More details in the [assignment spec](https://courses.cs.washington.edu/courses/cse579/26sp/projects/homework2/CSE579SP26_HW2.pdf).
- `main.py` (hyperparameter tuning only)
- `policy_gradient.py` — policy gradient TODOs
- `actor_critic.py` — actor-critic TODOs
- `sac.py` — SAC TODOs (extra credit)
