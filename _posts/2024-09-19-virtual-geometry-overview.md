# Rendering of high density geometry in Omniforce Engine, part #1
## The problem
The computer graphics industry was using the discrete levels of detail (LOD) for meshes in real-time rendering during its entire lifetime. However, it may be not as efficient as we would like it to be.

We can generate many LODs, to correctly "ammortize" the rendering cost, where each consecutive LOD will have *X* times less indices than previous level. However, it still scales poorly when we have to deal with large meshes in front of the camera. For example, we have a large spaceship mesh in the scene and we need to render it. If camera is close to the spaceship, we would need to render entire spaceship, even the parts which are far away in *full quality*, which is suboptimal. We need non-uniform level of detail to ammortize rendering cost even better. We need a system which allows us for rendering the same mesh with *non-uniform level of detail*.

Furthermore, the visibility culling of a geometry is a problem too. Why do we even need to render those parts of the mesh which are not visible in the camera due to various reasons? To solve it, we need break a mesh into small clusters which we would render and most importantly - *cull* individually. Those small clusters are also often called "meshlets" and I will use this word too.

To summarize:
- Traditional discrete LODs are suboptimal, no need to render distant clusters in the same LOD as we render close clusters
- Need to do per-cluster visibility culling
- When streaming, we can't have 50% of LOD 0 and LOD 1 to be resident due to discrete LODs

## The problem of per-cluster LODs
### Geometrical cracks
We have decided to go for per-cluster LODs, simplify not the entire mesh, but each clusters individually. However, it gives us a massive problem to solve.

If we generate a LOD per cluster, we would most certainly end up with a problem of geometrical *cracks* similar to the ones mentioned in the previous blog post but much more massive. It happens due to the fact that cluster boundaries most certainly will change because they are completely independent and might make different LOD decision, leading to the change of mesh's topology. As you might understand, it is completely unacceptable.

### Possible solution
To solve the problem with cracks, we could lock boundary edges of the clusters and only simplify their "interior". The boundary edges are those edges that are only "used" by 1 triangle.

However, it introduces another problem which is as bad as the problem above: we would end up with tons of locked edges that will never be simplified away and will remain in the mesh acroos *entire LOD chain*. It may seem to be not such a big problem, but due to this problem, we would barely be able to half the triangle count of a LOD, effectively eliminating the entire point of simplification.
