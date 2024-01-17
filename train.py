from diffusion_rs import Unet, GaussianDiffusion, Trainer

#print('linear')
num_classes = 1
#batch_size = 4
num_steps = 1000
model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=1,
        cond_drop_prob=0
    )

diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=num_steps,
        sampling_timesteps=25,
        beta_schedule = 'linear',#'cosine'，'linear'
        objective = 'pred_x0'
        
    )

trainer = Trainer(
    diffusion,
    'train_data',
    'test_data_n',
    augment_horizontal_flip = False,
    train_batch_size = 2,#64
    train_lr = 3e-4,#3e-4
    train_num_steps = 160000, #150000        # total training steps
    gradient_accumulate_every = 16,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid=False,                # whether to calculate fid during training
    num_samples = 16,
    save_and_sample_every=1,
    results_folder = './result_64_atten',
)
#trainer.load(28)
trainer.train()


print('aaaa')
