from after_test.trainer import Trainer
from after_test.config import get_config
from after_test.data_loader import get_loader
from after_test.data_loader import get_3d_loader
from after_test.utils import prepare_dirs_and_logger, save_config


def main(config):
    prepare_dirs_and_logger(config)
    loader = get_loader(config.data_dir, config.dataset, config.batch_size)
    loader_3d = get_3d_loader(config.batch_size)
    trainer = Trainer(config, loader_3d)
    save_config(config)
    trainer.train()


if __name__ == "__main__":
    print("main")
    config, unparsed = get_config()
    main(config)
