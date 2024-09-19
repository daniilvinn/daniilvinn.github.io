# Rendering of high density geometry in Omniforce Engine, part #1
## The problem
The computer graphics industry was using the discrete levels of detail (LOD) for meshes in real-time rendering during its entire lifetime. However, it may be not as efficient as we would like it to be.

We can generate many LODs, to correctly "ammortize" the rendering cost, where each consecutive LOD will have *X* times less indices than previous level. However, it still scales poorly when we have to deal with large meshes in front of the camera. For example, we have a large spaceship mesh in the scene and we need to render it. If camera is close to the spaceship, we would need to render entire spaceship, even the parts which are far away in *full quality*, which is suboptimal. We need non-uniform level of detail to ammortize rendering cost even better. We need a system which allows us for rendering the same mesh with *non-uniform level of detail*.

Furthermore, the visibility culling of a geometry is a problem too. Why do we even need to render those parts of the mesh which are not visible in the camera due to various reasons? To solve it, we need break a mesh into small clusters which we would render and most importantly - *cull* individually. Those small clusters are also often called "meshlets" and I will use this word too.

