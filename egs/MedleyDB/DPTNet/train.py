import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid import DPTNet
from asteroid.engine import schedulers

from asteroid.data.medleydb_dataset import SourceFolderDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch_audiomentations import Compose, Gain

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
#test_dir = "/data/EECS-Sandler-Lab/AcapellaDataset/split/tt/"
train_dir = "/jmain02/home/J2AD002/jxm06/sxs01-jxm06/data/split_5/tr/"
val_dir = "/jmain02/home/J2AD002/jxm06/sxs01-jxm06/data/split_5/cv/"

class AugSystem(System):
    def training_step(self, batch, batch_nb):
        mix, source = batch
        apply_augmentation = Compose(
            transforms=[
                Gain(
                    min_gain_in_db=-15.0,
                    max_gain_in_db=5.0,
                    p=0.5,
                    mode="per_channel"
                )
            ]
        )
        source = apply_augmentation(source, sample_rate=22050)
        loss = self.common_step((mix, source), batch_nb, train=True)
        self.log("loss", loss, logger=True)
        return loss    

def main(conf):
    exp_dir = conf["main_args"]["exp_dir"]
    # Define Dataloader
    """total_set = MedleydbDataset(
        conf["data"]["json_dir"],
        n_src=conf["data"]["n_inst"],
        n_poly=conf["data"]["n_poly"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        threshold=conf["data"]["threshold"],
    )
    """
    train_set = SourceFolderDataset(
        train_dir,
        train_dir,
        conf["data"]["n_poly"],
        conf["data"]["sample_rate"],
        conf["training"]["batch_size"],
        train = True
    )
    val_set = SourceFolderDataset(
        val_dir,
        val_dir,
        conf["data"]["n_poly"],
        conf["data"]["sample_rate"],
        conf["training"]["batch_size"],
    )
   
     
    train_loader = data.DataLoader(
        train_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    # Update number of source values (It depends on the task)
    conf["masknet"].update({"n_src": conf["data"]["n_inst"] * conf["data"]["n_poly"]})

    model = DPTNet(**conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"])
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    from asteroid.engine.schedulers import DPTNetScheduler

    schedulers = {
        "scheduler": DPTNetScheduler(
            optimizer, len(train_loader) // conf["training"]["batch_size"], 64
        ),
        "interval": "step",
    }

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = AugSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=schedulers,
        train_loader=train_loader,
        val_loader=val_loader,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
