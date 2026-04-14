from __future__ import annotations

import json

from grlnet import __version__
from grlnet.models import GRLNetWeights, grlnet_stabhrec40


def main() -> None:
    model = grlnet_stabhrec40(weights=None)
    num_params = sum(int(param.numel()) for param in model.parameters())
    print(
        json.dumps(
            {
                "package": "grlnet",
                "version": __version__,
                "default_model": "grlnet_stabhrec40",
                "num_params": num_params,
                "weights": sorted(GRLNetWeights.names()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
