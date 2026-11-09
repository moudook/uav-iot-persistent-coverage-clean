"""
The RNG discipline of an evaluation.

The simulator's own randomness is already local and seeded: the environment
holds a `numpy.random.default_rng(cfg.seed)`, so a scene is a function of the
seed and nothing else.  A POLICY is a different matter.  Eq. (13) of the
report specifies that the deployed controller draws,

    a_i(t) ~ pi_theta(a_i | o_i(t)),

so every deployed policy samples, and the sampling in `mappo.py` draws from
the GLOBAL torch RNG.  If that RNG is not seeded, then an evaluation is a
fresh random draw every time it is run: the same policy on the same scene
gives a different answer on every call.  Three things follow, and all three
are unacceptable in a paper whose claim is that every number is measured.

  1. The numbers cannot be reproduced from the stored artefact.  The same
     policy file, evaluated twice, gives two different answers.
  2. A comparison between two controllers carries an unquantified sampling
     component, because each controller is measured on a different draw.
  3. The spread reported for a controller mixes the scene-to-scene variation
     with the draw-to-draw variation, and the paper cannot say which is
     which.

`seed_evaluation` fixes all three.  The seed is derived from the evaluation
seed ALONE, so every controller is measured on the SAME random stream on the
SAME scene: the difference between two controllers is then a difference
between policies rather than between the luck of the draw (common random
numbers).  The numbers move a little when the discipline is introduced --
that is the size of the uncontrolled term that was there before -- and they
stop moving afterwards, which is the point.

The seed is deliberately far from the small integers, so that it cannot
collide with the training seeds, which use `torch.manual_seed(seed)`
directly.
"""

from __future__ import annotations

import numpy as np
import torch

# A large odd multiplier: the derived seeds are spread out, so that the
# evaluation of scene 0 does not sit next to the training seed 0 in the
# streams, and two scenes do not share a stream.
_MULT = 1_000_003


def seed_evaluation(seed: int) -> int:
    """Seed the global RNGs that a sampled policy draws from.

    Returns the derived torch seed, so that a caller can record which
    stream an evaluation used.
    """
    s = int(seed)
    t = _MULT * (s + 1) + 7
    torch.manual_seed(t)
    np.random.seed((t + 13) % (2 ** 31 - 1))
    return t
