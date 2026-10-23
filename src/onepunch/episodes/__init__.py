from onepunch.episodes.cv import family_folds, select_lambda0
from onepunch.episodes.loss import (episode_loss, episode_losses, fit_predict, validation_loss,
                                    standardize_scales)
from onepunch.episodes.sampler import EpisodeSampler, sample_step_pool

__all__ = ["EpisodeSampler", "sample_step_pool", "episode_loss", "episode_losses", "validation_loss",
           "standardize_scales", "fit_predict", "select_lambda0", "family_folds", "build_step_pool_cfg"]


def build_step_pool_cfg(cfg) -> dict | None:
    """Step-pool settings from the `episodes` config (None -> use all models every step)."""
    if not cfg.get("step_families"):
        return None
    return {"n_families": int(cfg["step_families"]), "models_per_family": int(cfg["step_models_per_family"]),
            "redraw_every": int(cfg.get("redraw_pool_every", 10))}
