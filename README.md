# Road Rage

Hate traffic? Me too. Traffic lights should be smarter and controlling
them is a juicy optimization problem. This experiment uses reinforcement
learning to dynamically control stop lights to optimize getting people
where they want to go.


## Usage


```
# Install sumo the traffic simulator
sudo apt-get install sumo sumo-tools sumo-doc
export $SUMO_HOME=/usr/share/sumo

# Setup and install requirements
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training and evaluation
python run.py
```


## Options

```
python run.py --help

Usage: run.py [options]

Options:
  -h, --help            show this help message and exit
  --gui                 Run the GUI version of sumo
  --type=TYPE           The type of prediction to use
  --training-steps=TRAINING_STEPS
                        The number of simulation steps to train for
  --training-max-steps=TRAINING_MAX_STEPS
                        The maximum number of steps during training per
                        episode
  --training-warmup=TRAINING_WARMUP
                        Steps to take randomly before prediction
  --eval-episodes=EVAL_EPISODES
                        Number of episodes to evaluate for
  --eval-max-steps=EVAL_MAX_STEPS
                        Max simulation steps per episode during training
```


## Results

Reward (higher is better):

- DQN (trained 200000 epochs): 13027
- random: 6965
- timed: 1495