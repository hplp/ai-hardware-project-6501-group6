# Final Project Report

## Team Name
6501-Group 6

## Team Members
- Peilin Chen
- Xinyuan Fu
- Hanyuan Gao
- Feilian Dai

## Project Title

Design Space Exploration for Compressed Deep Convolutional Neural Network on SCALE Sim

## Project Description

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

We select a typical and popular model **LeNet-5** for this project. The application is hand-written digit recognition.

### Training for LeNet-5 Neural Network

The LeNet-5 structure is shown below.

<img src="/figs/lenet5.jpg" alt="architecture" align="center" width="80%">

LeNet-5 NN has three convolutional layers (C1, C3, and C5), two pooling layers (S2 and S4), and two fully-connected layers (F6 and OUTPUT). 
The LeNet-5 parameter size can be calculated using the formula below:

C1(1*5*5*6+6) + C3(6*5*5*16+16) + C5(5*5*16*120+120) + F6(120*84) + OUTPUT(84*10) = 61,686

We can see that LeNet-5 is pretty small neural network with only 61K trainable parameters. We have finished the LeNet-5 training process. 
Please look at the lenet-5_int8_quant folder in this repository to find the training code. To give a clear instruction, we show the structure of this repo
below.

```
  / lenet-5_int8_quant /
             |--- data/
             |      
             |--- save_model/
             |                 |--- best_model.pth
             |                 |--- last_model.pth
             |                 \--- quant_model.pth
             |
             |--- weight/
             |          |--- *.bias.txt
             |          |--- *.weight.txt
             |          \--- *_scale_zero.txt
             |
             |--- net.py
             |
             |--- net_quant.py
             |
             |--- quant.py
             |
             |--- scale_shift.ipynb
             |           
             |--- test_quant.py
             |
             \--- train.py
```

The description for the subfolder: 

**data**: the training and test data (MNIST dataset) are stored in this folder. 

**save_model**: this folder saves the model during the training process and the quantized model. 

**weight**: we report the quantized (int8) weights and quantization-related parameters (scale factor and zero point) in this folder.

We first define the LeNet-5 neural network structure using Pytorch library in net.py. Then we train the LeNet-5 NN using MNIST dataset in train.py. The training and test batch size are 32 and 1000, respectively. We use the Cross-Entropy Loss function to measure the difference between the predicted probability distribution and the true labels. The optimizer we use is the Stochastic Gradient Descent (SGD). The learning rate and momentum are 0.001 and 0.9, respectively. The total training epoch is 
50. The training result is shown below.

<img src="/figs/train_loss_and_val_loss.jpg" alt="architecture" align="center" width="50%">

<img src="/figs/train_acc_and_val_acc.jpg" alt="architecture" align="center" width="50%">

The first figure depicts the training loss and validation loss during the 50 training epoches. The second figure depicts the training accuracy and validation
accuracy during the 50 training epoches. The best accuracy achieved by our trained LeNet-5 model is 98.56%.

### Quantization

We use linear asymmetric quantization method to quantize the LeNet-5 NN into 8bit. QuantLinear, QuantConv2d, and QuantAvePool2d functions 
are defined in net_quant.py. The quant.py will utilize the net_quant.py to quantize the trained LeNet-5 NN and save the int8 parameters in *.txt. 
Quantized parameters can be found in the **weight** folder. After finishing the training of LeNet-5, we can run quant.py program to quantize the model into
8bit. Then we can run the test_quant.py program to test the accuracy of quantized model. 

<img src="/figs/test_quant_model.jpg" alt="architecture" align="center" width="50%">

We test the accuracy of the quantized LeNet-5 model using the test MNIST dataset. The testing result is shown above. We can observe that compared to the original LeNet-5 model, the quantized int8 LeNet-5 model achieves no accuracy loss (even higher than the original model in floating point datatype).

## Hardware Part (Design Space Exploration)








## Conclusion
