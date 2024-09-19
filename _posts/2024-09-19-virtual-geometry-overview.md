# Rendering of high density geometry in Omniforce Engine, part #1
## The problem
The computer graphics industry was using the discrete levels of detail (LOD) for meshes in real-time rendering during its entire lifetime. However, it may be not as efficient as we would like it to be.

We can generate many LODs, to correctly "ammortize" the rendering cost, where each consecutive LOD will have *X* times less indices than previous level. However, it still scales poorly when we have to deal with large meshes in front of the camera. For example, we have a large spaceship mesh in the scene and we need to render it. If camera is close to the spaceship, we would need to render entire spaceship, even the parts which are far away in *full quality*, which is suboptimal. We need non-uniform level of detail to ammortize rendering cost even better. We need a system which allows us for rendering the same mesh with *non-uniform level of detail*.

Furthermore, the visibility culling of a geometry is a problem too. Why do we even need to render those parts of the mesh which are not visible in the camera due to various reasons? To solve it, we need break a mesh into small clusters which we would render and most importantly - *cull* individually. Those small clusters are also often called "meshlets" and I will use this word too.

The issue also comes when we think about memory. High density, film-quality meshes have *a lot* of vertices and indices, and of course, we need to store them somewhere in the VRAM, and it is not free. Even with mesh compression system mentioned in previous post, the memory footprint is still very high. To solve it, we would need streaming, but the next question arises - why do we even need to keep those clusters that are not rendered in current frame? Again, with per-cluster LODs we can render mesh with non-uniform quality, meaning that we can "combine" multiple LODs at once. So, we need a streaming system that allows for partial LOD residency.

To summarize:
- Traditional discrete LODs are suboptimal, no need to render distant clusters in the same LOD as we render close clusters
- Need to do per-cluster visibility culling
- When streaming, we can't have X percents of LOD Y and LOD Z to be resident due to discrete LODs

## The problem of per-cluster LODs
### Geometrical cracks
We have decided to go for per-cluster LODs, simplify not the entire mesh, but each clusters individually. However, it gives us a massive problem to solve.

If we generate a LOD per cluster, we would most certainly end up with a problem of geometrical *cracks* similar to the ones mentioned in the previous blog post but much more massive. It happens due to the fact that cluster boundaries most certainly will change because they are completely independent and might make different LOD decision, leading to the change of mesh's topology. As you might understand, it is completely unacceptable.

### Possible solution
To solve the problem with cracks, we could lock boundary edges of the clusters and only simplify their "interior". The boundary edges are those edges that are only "used" by 1 triangle.

However, it introduces another problem which is as bad as the problem above: we would end up with tons of locked edges that will never be simplified away and will remain in the mesh acroos *entire LOD chain*. It may seem to be not such a big problem, but due to this problem, we would barely be able to half the triangle count of a LOD, effectively eliminating the entire point of simplification.

To sum up:
- Can not lock all clusters' edges - ends up with really poor simplification
- Neither can unlock all edges - leads to massive topology changes and geometrical cracks

## The idea
### Summarization
Even considering all the problems above, **the problem is solvable** - I wouldn't write this post if it wasn't possible.

Let's revise all requirements for the system:
1. Need clusterized meshes to allow for high granularity visibility culling
2. More importantly - need per-cluster levels of detail
3. Can not accept geometrical cracks - need to lock edges
4. Can not afford locking *all* edges - leads to extreme amount of persistently locked vertices

The #4 statement is the most important. The system can not afford locking *all edges*, but can afford locking *part* of the edges. So how to do we evaluate which vertices need to lock?

### Grouping clusters
If we can't "communicate" between *independent* clusters, so they make the same LOD decision, we *force* them to do so by grouping them together, making them no longer independent.

We group clusters by some factor (see below), then merge them together into one, lock edges, simplify and split back to clusters. This way can group 4 clusters together, merge them, simplify, but get 2 clusters as output *while still preserving the group boundaries*. See the photo below that shows how it looks like. On the photo, the group above has 4 clusters, 4 triangles each, and then it gets simplified and split back to two meshlets that *preserve the boundaries*.
![Showcase of "Group, merge, lock, simplify, split back" process](https://github.com/user-attachments/assets/e4f35374-514f-4fdb-b344-6f62ee1afdb1)

"The grouping factor" mention above is number of shared edges between clusters. We want to simplify as much as possible to reach 1/2 index count, therefore we want to lock as small amount of edges as possible. All these tips mean that we need to group those clusters which have the most shared edges, because those edges will not be locked. Grouping is a common graph partitioning problem, where graph vertices (nodes) are clusters, edges are the fact that two clusters are connected and edge weights are number of actual shared mesh edges between two clusters.

After splitting groups back, we add it to a cluster pool from which new clusters are generated. The pool is cleared after every simplification pass.

After all that, we will end up with a DAG (Directed Acyclic Graph) or basically a hierarchy of clusters, where root of the graph is lowest detail level with ideally 1 cluster. The next question is: "how do we traverse, select and render those clusters?"

### The hierarchy
Due to the fact that merged groups are split back, we still cannot render all clusters independently - we can only render entire groups to avoid cluster overlap and / or missing clusters that will lead to holes in the mesh. It may sound bad, but groups are really small and their cluster count depends on an implementation and varies from 4 to 8. Average high-density mesh may have thousands of clusters at LOD 0, so we have ~1-3 thousands of groups to render at LOD 0. So, if we LOD test cluster *X* and it fails (too detailed / too low detailed), we need to test all of its children groups and repeat it until we find a group that perfectly fits. What "fits" and what doesn't will be discussed below. You could also think that if we can only render entire groups, then we do not need to test all meshlets individually - we only need to test groups. The "selection" of groups form a *DAG cut*. See the photo below that shows how it looks like.

**Note**: the error **must** be in world-space in order to correctly LOD test groups.

![The DAG cut](https://github.com/user-attachments/assets/0601934c-20cc-4e57-b02a-0904c1f3f3a0)

### How to render
After mesh build we end up with a hierarchy of clusters, where parents are simplified versions of their children. The traversal always starts from root (least detailed LOD), assuming that most meshes will be rendered in low quality. During the build, each groups "gets" a bounding sphere which encloses all vertices inside the group and the simplification error **and** its parent bounding sphere and error to allow entirely local (we only need group X data to test if it fits), parallel LOD selection - this is one of the most important parts.

### LOD test
We only render a group if it is detailed enough to not produce any perceptible error **and** its parent is simplified too much. The error is perceptible only if its projected sphere's radius is equal or more than 1 pixel. If it less than 1 pixel, then the error is imperceptible.

Now we need to understand what is "projected sphere". When testing a LOD, we form a sphere with center at group's bounding sphere and radius that equals to the error, then we project it to the screen with current view parameters and get size of the sphere in pixels - a metric used in LOD testing.

Important note: whenever a group with just enough error is found, we stop its subtree traversal because all of its children LOD tests will fail due to the fact that the children and children's children will be too detailed to be rendered).

