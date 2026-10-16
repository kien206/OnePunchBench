"""Train: search the prompt p* and fit the readout W* (configs/README.md explains every setting).

    python scripts/describe_run.py pool=tier_s                 # check models / roles / benchmarks first
    python train.py                                            # smoke pool, fixed prompt (search=none)
    python train.py search=random_swap search.loop.steps=200   # gradient-free prompt search
    python train.py pool=tier_s search=random_swap seed=1 logger.mode=offline
    python train.py -m seed=0,1,2                              # Hydra multirun

Writes runs/<date>/<time>_train/artifact/ (prompt.txt, predictor.npz, config.yaml); use it with predict.py.
"""

import hydra
from omegaconf import DictConfig

from onepunch.pipeline import train


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
