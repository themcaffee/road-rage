# Traffic Rage

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
