# Tiliter ML

This repository contains my results for the tiliter ML challenge

## Setup dataset

I've ignored the dataset in this repo since it is best practice to store data in an object store of some kind. To setup
the dataset, unzip the contents to `data/`:

```bash
cd tiliter-ml
unzip digits.zip

unzip flowers.zip
mv flowers data/

rm flowers.zip digits.zip
```

## Environment

Create a virtual environment for this project:

```bash
conda create -n tiliter-ml python=3.6
```

Freeze the enviroment to requirements file:

```bash
pip freeze > requirements.txt
```

## Build

To restore the environment from requirements file

```bash
pip install -r requirements.txt
```

## Run

The `train.py` script executes the training run. The run may be configured by editing the parameters in the `CONFIG` variable.

### Mnist

The existing parameters train the model with an accuracy of `0.9901` on the test set.

```bash
python train.py --config-id mnist
```

### Flowers

The existing parameters train the model for 11 epochs until we start to overfit with an accuracy of `0.7002` on the test set.

```bash
python train.py --config-id flowers
```
