# Rain demo: AI learns to avoid raindrops in a simple game

Rain is a simple game where an AI (X) must learn to avoid raindrops (O) which randomly fall from the sky.

```
    O
  O
O
  X
```

## Requirements

```sh
# Install requirements with pip.
pip install --user -r requirements.txt
```

## Instructions

```sh
# 1) Watch the AI play.
python train.py -v -d easy

# 2) Train the AI.
python train.py

# 3) Return to step 1 and see if the AI improved.
```

## Advanced

```sh
# Change the difficulty with "-d easy".
python train.py -d easy

# Clear the AI's memory with "-r 0".
python trains.py -r 0

# Watch the AI train with "-v".
python train.py -v
```

## Notes about easy mode vs. hard mode
It's much faster to train the AI in hard mode, where the AI gets instant, accurate feedback. From the first move, a single mistake means a loss. In easy mode, it's easier for bad behavior to be accidentally reinforced.
