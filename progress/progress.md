
## Progress

### Peilin Chen:

I have finished the workload I am responsible for. I select a typical and popular model **LeNet-5** for this project. 

#### lenet-5_int8_quant folder structure
```
  / lenet-5_int8_quant /
             |--- data/
             |      
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
             |
             |--- net_quant.py
             |
             |
             |--- scale_shift.ipynb
             |           
             |
             |--- test_quant.py
             |
             |
             \--- train.py
```

