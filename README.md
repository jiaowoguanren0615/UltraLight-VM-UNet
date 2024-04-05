# [UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation](https://arxiv.org/abs/2403.20035)
This is a warehouse for UltraLight-VMUNet-Pytorch-model, can be used to train your image-datasets for segmentation tasks.
The code mainly comes from official [source code](https://github.com/wurenkai/UltraLight-VM-UNet)

## Install mamba_block & causal-conv1d
click [here](https://github.com/jiaowoguanren0615/Install_Mamba/)

## Project Structure
```
├── datasets: Load datasets
    ├── mydataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── Prepare_ISIC2017.py: Prepare ISIC2017 dataset
    ├── Prepare_ISIC2018.py: Prepare ISIC2018 dataset
    ├── Prepare_PH2.py: Prepare PH2 dataset
    ├── Prepare_your_dataset.py: Prepare your own dataset
├── models: UltraLight VM-UNet Model
    ├── build_model.py: Construct "UltraLight VM-UNet" model
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: BCEDiceLoss
    ├── lr_scheduler.py: create a lr_scheduler based on LambdaLR 
    ├── optimizer.py: Define Sophia optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
└── train_gpu.py: Training model startup file
```

## Precautions
Please run ___Prepare_ISIC2017/2018/PH2.py___ first before training corresponding dataset.
If you want to train your own data set. Please run ___Prepare_your_dataset.py___ firstly. Additionally, enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters. 

## Use Sophia Optimizer (in util/optimizer.py)
You can use anther optimizer sophia, just need to change the optimizer in ___train_gpu.py___, for this training sample, can achieve better results
```
optimizer = SophiaG(model.parameters(), lr=args.lr, betas=(0.965, 0.99), rho=0.01, weight_decay=args.weight_decay)
```

## Train this model
### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. 

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{wu2024ultralight,
  title={UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation},
  author={Wu, Renkai and Liu, Yinghao and Liang, Pengchen and Chang, Qing},
  journal={arXiv preprint arXiv:2403.20035},
  year={2024}
}
```
