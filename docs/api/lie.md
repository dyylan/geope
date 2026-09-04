# geope.geometry.lie

The Lie-algebra and Lie-group utilities the pulse model is built on. The group
*manifolds* live on the [Manifolds](manifolds.md) page.

::: geope.geometry.lie.Basis

## The coefficient projector

Resolving a batch of ambient matrices against the frame,
$c_i = \mathrm{Re}\,\mathrm{Tr}(G_i X)/d$. Above 5 qubits the on-the-fly projector
`get_project_omegas_fn_otf` builds the Pauli strings on the fly to avoid
materialising the full $(K, d, d)$ frame.

::: geope.geometry.lie.basis.project_omegas

::: geope.geometry.lie.basis.get_project_omegas_fn

::: geope.geometry.lie.basis.get_kron_chain

::: geope.geometry.lie.basis.get_project_omegas_fn_otf
