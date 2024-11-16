import torch
from torch import nn
from net_quant import LeNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet().to(device)
model.load_state_dict(torch.load("./save_model/best_model.pth"))

model.linear_quant()

folder = './weight/'
for name in model.state_dict():
    file = open(folder + name + ".txt", "w")
    file.write(str(model.state_dict()[name]))
    file.close()

file = open(folder + "c1_scale_zero.txt", "w")
file.write(str(model.c1.scale))
file.write("\n" + str(model.c1.zero_point))
file.close()

file = open(folder + "c3_scale_zero.txt", "w")
file.write(str(model.c3.scale))
file.write("\n" + str(model.c3.zero_point))
file.close()

file = open(folder + "c5_scale_zero.txt", "w")
file.write(str(model.c5.scale))
file.write("\n" + str(model.c5.zero_point))
file.close()

file = open(folder + "f6_scale_zero.txt", "w")
file.write(str(model.f6.scale))
file.write("\n" + str(model.f6.zero_point))
file.close()

file = open(folder + "output_scale_zero.txt", "w")
file.write(str(model.output.scale))
file.write("\n" + str(model.output.zero_point))
file.close()

torch.save(model.state_dict(), "./save_model/quant_model.pth")
