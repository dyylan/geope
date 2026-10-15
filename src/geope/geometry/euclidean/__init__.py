r"""Euclidean motion groups: rigid transforms, which are not $\mathrm U(N)$-orbits.

* `SpecialEuclideanGroup` — $\mathrm{SE}(3)$ under the product metric of
  $\mathrm{SO}(3)\times\mathbb R^3$ with a length scale $\ell$, the configuration
  space of a rigid body and of a serial arm's tool frame (see `geope.robotics`).
"""

from .se3 import SpecialEuclideanGroup, hat, so3_log_vee, vee
