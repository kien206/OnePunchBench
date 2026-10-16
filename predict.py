"""Inference: predict the benchmark scores of any model from one forward pass over p*.

    python predict.py predict.artifact=runs/<date>/<time>_train/artifact predict.model_ids=[Qwen/Qwen2.5-0.5B]
    python predict.py predict.artifact=... pool=tier_s predict.splits=[testA]   # adds metrics where scores exist

Writes runs/<date>/<time>_predict/predictions.csv (and metrics.json when ground truth is known).
"""

import hydra
from omegaconf import DictConfig

from onepunch.pipeline import predict


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(predict(cfg).to_string(index=False))


if __name__ == "__main__":
    main()
