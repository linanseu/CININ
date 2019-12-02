This folder contains the codes for each cluster devices and the central node.

For example, assuming you have a host machine H, gateway device G, and two edge devices E0 (data source) and E1 (idle), while you want to perform a 5x5 FTP with 16 fused layers, then you need to follow the steps below:

In gateway device G:

./deepthings -mode gateway -total_edge 2 -n 5 -m 5 -l 16
In edge device E0:

./deepthings -mode data_src -edge_id 0 -n 5 -m 5 -l 16
In edge device E1:

./deepthings -mode non_data_src -edge_id 1 -n 5 -m 5 -l 16
Now all the devices will wait for a trigger signal to start. You can simply do that in your host machine H:

./deepthings -mode start
