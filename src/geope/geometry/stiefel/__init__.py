"""The Stiefel family: manifolds of orthonormal frames, which are not Lie groups.

$\\mathrm{St}(n, p) = \\{X \\in \\mathbb C^{n \\times p} : X^\\dagger X = I\\}$ is a
homogeneous space $\\mathrm U(n)/\\mathrm U(n-p)$, so it has no identity, no left
translation and hence no global trivialisation: its tangent spaces are genuinely
point-dependent. Everything here therefore goes through the point-based
`geope.geometry.manifold.Manifold` interface rather than the Lie-group
specialisation in `geope.geometry.lie`.

* `Stiefel` — the general $m$-frame under the canonical metric, with the
  iterative Zimmermann-Hueper logarithm.
* `StateSphere` — the $p = 1$ case as $\\mathbb{CP}^{n-1}$, where the logarithm
  is a closed-form great circle. Prefer it for state preparation.
"""

from .sphere import StateSphere
from .stiefel import Stiefel
