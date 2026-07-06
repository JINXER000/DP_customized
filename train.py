"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import logging
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import torch


class _EGLProbeSpamFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return (
            ("is available for rendering" not in msg)
            and ("is not available for rendering" not in msg)
            and ("egl_probe/build/test_device" not in msg)
        )


def _suppress_egl_probe_spam() -> None:
    root_logger = logging.getLogger()
    has_filter = any(
        isinstance(log_filter, _EGLProbeSpamFilter)
        for log_filter in root_logger.filters
    )
    if not has_filter:
        root_logger.addFilter(_EGLProbeSpamFilter())

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    _suppress_egl_probe_spam()

    DEVICE = cfg.training.device
    # Set device for older PyTorch versions
    if DEVICE.startswith('cuda'):
        torch.cuda.set_device(DEVICE)
    # For CPU, no device setting needed

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
