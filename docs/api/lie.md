# geope.geometry.basis

The frame the pulse model is built on, and the projector that resolves against
it. Despite the Pauli-algebra flavour this is **not** Lie-specific — it is the
*ambient* coefficient frame, which both Stiefel manifolds take too — so it lives
at the root of `geope.geometry` rather than under `lie`. The group *manifolds*
live on the [Manifolds](manifolds.md) page.

::: geope.geometry.basis.Basis

## The coefficient projector

Resolving a batch of ambient matrices against the frame,
$c_i = \mathrm{Re}\,\mathrm{Tr}(G_i X)/d$. Above 5 qubits the on-the-fly projector
`get_project_omegas_fn_otf` builds the Pauli strings on the fly to avoid
materialising the full $(K, d, d)$ frame.

::: geope.geometry.basis.project_omegas

::: geope.geometry.basis.get_project_omegas_fn

::: geope.geometry.basis.get_kron_chain

::: geope.geometry.basis.get_project_omegas_fn_otf
