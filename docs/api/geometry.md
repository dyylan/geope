# geope.geometry

The geometry layer owns the manifold GEOPE walks, its tangent bundle, and every
per-step geometric quantity the optimisers read. The ownership chain is

```
Manifold  --owns-->  compute_point      (the chart Phi: parameters -> the manifold)
   |                 target         (the point being synthesised)
   |
   '-----owns-->  TangentBundle --owns-->  Basis      (coordinatises the fibres)
                                          jacobian   (the pushforward D Phi)
                                          hvp        (the second differential D^2 Phi)
```

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

## Binding

::: geope.geometry.binding.bind_manifold
