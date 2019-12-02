# Distribued Neural Network on Edge Devices

This repo contains the implementation of CININ: an distributed CNN inference framework over the edge clusters.
CININ mainly consists of:

CNN progressive retraining which takes the original trained CNN model as the input, and apply the changes on CNN models for efficient parallel inference over the cluster devices.

A distributed runtime system for device clusters.

## Progressive retraining

1. Download the caltech101 or imagenet dataset, as well as the pretrain model in the current folder. 
 
2. preprocess01.py and preprocess02.py are designed to preprocess the dataset, use ```python3 preprocess01.py``` to generate *label.txt* and use ```python3 preprocess02.py``` to generate *dataset-train.txt* and *dataset-test.txt*.

3. To run the code, type the following command

```
python3 main.py
```
To see the definitions of all the parameters, type the following command:

```python3 main.py --help``` 

