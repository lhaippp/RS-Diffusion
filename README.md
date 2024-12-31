# RS-Diffusion

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
You can check the state of a diffusion model during training by:
```
python sample_RS_real.py --config config/RS_real_config_teat.yaml
```
