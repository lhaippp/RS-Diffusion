import argparse
import yaml
from model.diffusion_model import Unet, RS_Diffusion, Trainer
from dataset.RS_Real_dataset import RS_Real_Train_dataset, RS_Real_test_dataset
import pathlib

def main(config_path):
    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize the model
    model = Unet(
        dim=config["model"]["dim"],
        dim_mults=tuple(config["model"]["dim_mults"]),
        num_classes=config["model"]["num_classes"],
        cond_drop_prob=config["model"]["cond_drop_prob"],
    )

    # Initialize the diffusion model
    diffusion = RS_Diffusion(
        model,
        image_size=config["diffusion"]["image_size"],
        timesteps=config["diffusion"]["timesteps"],
        sampling_timesteps=config["diffusion"]["sampling_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        objective=config["diffusion"]["objective"],
    )

    # Initialize datasets
    train_dataset = RS_Real_Train_dataset(
        folder=config["dataset"]["train_folder"],
        image_size=config["dataset"]["image_size"],
        augment_horizontal_flip=config["dataset"]["augment_horizontal_flip"],
    )
    test_dataset = RS_Real_test_dataset(
        folder=config["dataset"]["test_folder"],
        image_size=config["dataset"]["image_size"],
        augment_horizontal_flip=config["dataset"]["augment_horizontal_flip"],
    )

    # Initialize the trainer
    trainer = Trainer(
        diffusion,
        train_dataset,
        test_dataset,
        augment_horizontal_flip=config["dataset"]["augment_horizontal_flip"],
        train_batch_size=config["trainer"]["train_batch_size"],
        test_batch_size= config["trainer"]["test_batch_size"],
        train_lr=config["trainer"]["train_lr"],
        train_num_steps=config["trainer"]["train_num_steps"],
        gradient_accumulate_every=config["trainer"]["gradient_accumulate_every"],
        ema_decay=config["trainer"]["ema_decay"],
        amp=config["trainer"]["amp"],
        save_and_sample_every=config["trainer"]["save_and_sample_every"],
        results_folder=config["trainer"]["results_folder"],
        log_path=config["trainer"]["log_path"],
    )

    # Start training
    trainer.load(config["test"]["checkpoint"])
    trainer.sample(pathlib.Path(config["test"]["save_path"]),'cuda')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model using a configuration file.")
    parser.add_argument(
        "--config", 
        type=str, 
        default='config/RS_real_config_teat.yaml', 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
