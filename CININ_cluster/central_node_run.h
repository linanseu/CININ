#ifndef GATEWAY_H
#defineGATEWAY_H
#include "darkiot.h"
#include "configure.h"
void work_stealing_thread(void *arg);
device_ctxt* gateway_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t total_edge_number, const char** addr_list);
void gateway(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t total_edge_number, const char** addr_list);
#endif
