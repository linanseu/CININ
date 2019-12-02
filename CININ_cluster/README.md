##This folder contains the codes for each cluster devices and the central node.

central_node_run.c: The process run at the central node, which runs the tile scheduling algorithm, input partition and assignment as described in the paper.

cluster_run.c: The process run at each cluster devices, it takes the input tiles and performs the distributed CNN inference, and returns the intermediate results to the central node.

cnn_with_comm.py: contains the CNN model run at each cluster device

statistics_collection.c: statistics collection at each cluster devices, as described in the paper.
