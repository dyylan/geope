# geope.geometry

The geometry layer owns the space GEOPE walks, its tangent bundle, and every
per-step geometric quantity the optimisers read.

## One ambient space, several submanifolds

Every manifold in the library is a submanifold of one **ambient space**
$\mathcal A = \mathbb C^{N\times m}$, and each is the orbit of the same
$\mathrm U(N)$ action through a different base point:

| base point $x_0$ | the orbit through it |
| --- | --- |
| $\mathbb 1_N$ | $\mathrm U(N)$, $\mathrm{SU}(N)$ |
| $(\mathbb 1_m;\,0)^\intercal$ | $\mathrm{St}_m(\mathbb C^N)$ |
| $\lvert\psi_0\rangle$ | $S^{2N-1} \to \mathbb{CP}^{N-1}$ |

The pulse acts on all of $\mathcal A$ the same way, so everything *valued in*
$\mathcal A$ is shared, and everything happening *inside a fibre*
$T_x\mathcal M$ belongs to the submanifold. That single line is the layering:

```
ambient  A = C^{N×m}     Φ(φ) = U(φ)·x₀ ,  DΦ ,  D²Φ            chart.py
                         the coefficient frame
     │
     │   to_tangent :  A → T_x M         <- the only bridge
     ▼
submanifold  M ⊂ A       inner, coefficients, log,
                         tangent_acceleration,
                         hessian_quadratic_form, fidelity        Manifold hooks
```

## Ownership

```
Manifold  --owns-->  base_point     (x₀ = Φ(0): where the orbit starts)
   |                 target         (the point being synthesised)
   |                 compute_point  (the chart Φ: parameters -> the manifold)
   |
   '-----owns-->  TangentBundle --owns-->  frame      (the ambient coefficient frame)
                                          jacobian   (the pushforward DΦ)
                                          hvp        (the second differential D²Φ)
```

A manifold contributes only the point its orbit starts from; the pulse model
supplies the rest. `Manifold.bind` composes the two and holds no mathematics of
its own.

Nothing above this layer knows what space it is optimising on: `Geope`, the line
searches and `Gecko` speak only to the interface below. A manifold that is not a
Lie group — no identity, no left translation, point-dependent tangent spaces —
satisfies exactly the same contract; `geope.geometry.stiefel.sphere.StateSphere`
is the worked example.

## The interface

::: geope.geometry.manifold.Manifold

## The tangent bundle

::: geope.geometry.tangent.TangentBundle

## The per-step context

::: geope.geometry.context.GeometricContext
