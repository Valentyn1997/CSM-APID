import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from lightning_fabric.utilities.seed import seed_everything


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of train_data_dict
    torch.set_default_device(args.exp.device)
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)

    data_dict = dataset.get_data()
    model = instantiate(args.model, args, _recursive_=True)

    f_dict = {'Y_f': torch.tensor([[args.dataset.Y_f]]), 'T_f': args.dataset.T_f}

    # try:
    model.fit(train_data_dict=data_dict, f_dict=f_dict, log=args.exp.logging)
    # except RuntimeError:
    #     pass

    model.mlflow_logger.experiment.set_terminated(model.mlflow_logger.run_id) if args.exp.logging else None


if __name__ == "__main__":
    main()
