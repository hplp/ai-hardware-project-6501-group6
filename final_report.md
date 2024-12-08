# Final Project Report

## Team Name: 
6501-Group 6

## Team Members:
- Peilin Chen
- Xinyuan Fu
- Hanyuan Gao
- Feilian Dai

## Project Title:

Design Space Exploration for Compressed Deep Convolutional Neural Network on SCALE Sim

## Project Description:

Quantize convolutional neural network (convolutional layer and feedforward layer) into low-bit and deploy it on the open source 
platform Systolic CNN AcceLErator Simulator (SCALE Sim) using the best hardware configurations obtained by our design 
space exploration method. 


## SCALE Sim

SCALE Sim is a simulator for systolic array based accelerators for neural network layers that use General Matrix Multiplications (GEMMs). 
The overall architecture is shown below.

<img src="/figs/scalesim-overview.png" alt="architecture" align="center" width="80%">

SCALE Sim takes two files as input from the users. One is the hardware configuration, including systolic array height and width, input feature map (IFMAP)
SRAM size, filter SRAM size, output feature map (OFMAP) SRAM size, and dataflow. The other is the convolutional neural network topology (workload). SCALE Sim 
generates two types of outputs. One is the cycle-accurate traces for SRAM and DRAM. The other is the metrics like cycle counts, hardware utilization, bandwidth  
requirement, and total data movement, etc.

The SCALE Sim supports the computation of 8-bit GEMM. So we first try to quantize the CNN into 8-bit (activation: 8-bit, weight: 8-bit). Due to the fact that 
it is difficult to achieve the best inference performance for a specific CNN given the fixed hardware, we plan to explore the optimal SCALE Sim hardware 
configuration for different CNNs (that is design space exploration).

## Software Part (Quantization)

We select a typical and popular model **LeNet-5** for this project.

### LeNet-5 Neural Network


### Quantization Method


## Hardware Part (Design Space Exploration)




