# Manifolds

The spaces GEOPE can synthesise on. Each implements the hooks of
`geope.geometry.manifold.Manifold`; see that page for the contracts they must
satisfy.

## Matrix Lie groups

`MatrixLieGroup` is the middle layer that supplies everything
left-trivialisation makes free — a point-independent metric and coefficient map,
a matrix-logarithm geodesic, and the product-of-exponentials chart with its
manual propagator derivatives.

::: geope.geometry.lie.groups.MatrixLieGroup

::: geope.geometry.lie.groups.SpecialUnitaryGroup

::: geope.geometry.lie.groups.UnitaryGroup

## The Stiefel family

Manifolds of orthonormal frames, $\mathrm{St}_m(\mathbb{C}^N) = \{Q \in
\mathbb{C}^{N\times m} : Q^\dagger Q = \mathbb 1_m\}$. These are homogeneous
spaces $\mathrm{U}(N)/\mathrm{U}(N-m)$, **not** groups, so they implement the
point-based interface directly.

They are what to synthesise on when only an $m$-dimensional subspace of the
target matters — a gate mediated by a bosonic mode, a subspace encoding, a state
preparation. The rest of the unitary is redundancy, and quotienting it out is
cheaper than optimising over it.

::: geope.geometry.stiefel.stiefel.Stiefel

::: geope.geometry.stiefel.sphere.StateSphere
