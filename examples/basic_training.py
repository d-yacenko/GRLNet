"""Minimal ImageNet-style GRLNet training command.

For full A100/Slurm runs, start from:
src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml
"""

from grlnet.recipes.imagenet.train import main


if __name__ == "__main__":
    main()
