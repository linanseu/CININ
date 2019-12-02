  
#ifndef EDGE_H
#define EDGE_H
#include "darkiot.h"
#include "configure.h"
device_ctxt* edge_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id);
void stealer_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id);
void victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id);
void partition_frame_and_perform_inference_thread(void *arg);
void steal_partition_and_perform_inference_thread(void *arg);
void serve_stealing_thread(void *arg);
#endif
