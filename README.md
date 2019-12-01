## CININ implementation

This is the github repo for the work "Fast Distributed Convolutional Neural NetworkInferencing with CININ".

### Progressive retraining of discributed DNN
1. Download caltech 101 dataset and imagenet dataset
2. preprocess01.py and preprocess02.py are used to preprocess the dataset, use ```python3 preprocess01.py``` to generate *label.txt* and use ```python3 preprocess02.py``` to generate *dataset-train.txt* and *dataset-test.txt*.
3. To run the retraining, use ```python3 main.py```. Use ```python3 main.py --help``` to see a list of default parameters.
  
### Run the code

1. To run the code, use the following command: 

```
python3 src/main.py --config=xxx_xxx --env-config=sc2 with env_args.map_name=xxx
```
--config can be one of the following four options: vdn_6h_vs_8z,vdn_corridor,qmix_6h_vs_8z,qmix_corridor (corridor is 6z_vs_24zerg scenario). For example 'vdn_6h_vs_8z' means 6h_vs_8z map with VDN as the mixing network.

--env_args.map_name can be one of the following two options:6h_vs_8z,corridor (corridor is the 6z_vs_24zerg scenario)

2. All the hyperparameters can be found at:  src/config/default.yaml, src/config/algs/*.yaml and src/config/envs/*.yaml

3. The test accuracy will be saved in the 'xxx_accuracy_list.txt', where xxx is the local_results_path parameter in default.yaml.

4. Communication overhead \beta will be saved in the 'xxx_comm_overhead.txt', where xxx is the local_results_path parameter in default.yaml.
