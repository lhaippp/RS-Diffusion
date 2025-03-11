# RS-Diffusion
[https://arxiv.org/abs/2407.02906](https://arxiv.org/abs/2407.02906)
### Dataset

Download the **RS-Real** dataset from [HuggingFace](https://huggingface.co/datasets/Yzl-code/RS-Diffusion) 

### Training
You can try training your own diffusion model using the following command:  
```
python train_RS_real.py --config config/RS_real_config.yaml
```

### Multi-GPU Training
The trainer has been equipped with [🤗Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator), please refer to the link if you are not yet familiar with it.

Multi-GPU training can be simply set up by using the following commands:
```
# First, configure the Accelerator on your own machine
accelerate config

# Then, launch your training
accelerate launch train_RS_real.py --config config/RS_real_config.yaml
```

### Testing
Download the Pre-trained Weights for **RS-Real** dataset from [HuggingFace](https://huggingface.co/Lhaippp/RS-Diffusion) and put it to `RS-Diffusion/checkpoint` and run the following command for testing.

```
python sample_RS_real.py --config config/RS_real_config_teat.yaml
```
