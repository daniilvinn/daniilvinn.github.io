# High density geometry rendering in Omniforce Engine, part #1
## The problem
The computer graphics industry has used the discrete levels of detail (LOD) for meshes in real-time rendering during its entire lifetime. However, it may not be as efficient as we would like it to be.

We can generate many LODs, to correctly "amortize" the rendering cost, where each consecutive LOD will have *X* times fewer indices than the previous level. However, it still scales poorly when we have to deal with large meshes in front of the camera. For example, we have a large spaceship mesh in the scene and we need to render it. If the camera is close to the spaceship, we would need to render the entire spaceship, even the parts that are far away in *full quality*, which is suboptimal. We need a non-uniform level of detail to amortize rendering cost even better. We need a system that allows us to render the same mesh with a *non-uniform level of detail*.

Furthermore, the visibility culling of a geometry is a problem too. Why do we even need to render those parts of the mesh that are not visible in the camera due to various reasons? To solve it, we need to break a mesh into small clusters which we would render and most importantly - *cull* individually. Those small clusters are also often called "meshlets" and I will use this word too.

The issue also comes when we think about memory. High-density, film-quality meshes have *a lot* of vertices and indices, and of course, we need to store them somewhere in the VRAM, and it is not free. Even with the mesh compression system mentioned in the previous post, the memory footprint is still very high. To solve it, we would need streaming, but the next question arises - why do we even need to keep those clusters that are not rendered in the current frame? Again, with per-cluster LODs we can render a mesh with non-uniform quality, meaning that we can "combine" multiple LODs at once. So, we need a streaming system that allows for partial LOD residency.

To summarize:
- Traditional discrete LODs are suboptimal, no need to render distant clusters in the same LOD as we render close clusters
- Need to do per-cluster visibility culling
- When streaming, we can't have X percent of LOD Y and LOD Z to be resident due to discrete LODs

## The problem of per-cluster LODs
### Geometrical cracks
We have decided to go for per-cluster LODs, simplifying not the entire mesh, but each cluster individually. However, it gives us a massive problem to solve.

If we generate a LOD per cluster, we would most certainly end up with a problem of geometrical *cracks* similar to the ones mentioned in the previous blog post but much more massive. It happens due to the fact that cluster boundaries most certainly will change because they are completely independent and might make different LOD decisions, leading to the change of mesh's topology. As you might understand, it is completely unacceptable.

### Possible solution
To solve the problem with cracks, we could lock the boundary edges of the clusters and only simplify their "interior". The boundary edges are those edges that are only "used" by 1 triangle.

However, it introduces another problem that is as bad as the problem above: we would end up with tons of locked edges that will never be simplified away and will remain in the mesh across *the entire LOD chain*. It may not seem to be such a big problem, but due to this problem, we would barely be able to half the triangle count of a LOD, effectively eliminating the entire point of simplification.

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
4. Can not afford to lock *all* edges - leads to extreme amount of persistently locked vertices

The #4 statement is the most important. The system can not afford to lock *all edges*, but can afford to lock *part* of the edges. So how do we evaluate which vertices need to lock?

### Grouping clusters
If we can't "communicate" between *independent* clusters, so they make the same LOD decision, we *force* them to do so by grouping them together, making them no longer independent.

We group clusters by some factor (see below), then merge them together into one, lock edges, simplify, and split back into clusters. This way can group 4 clusters together, merge them, and simplify, but get 2 clusters as output *while still preserving the group boundaries*. See the photo below that shows what it looks like. In the photo, the group above has 4 clusters, 4 triangles each, and then it gets simplified and split back to two meshlets that *preserve the boundaries*.
![Showcase of "Group, merge, lock, simplify, split back" process](https://github.com/user-attachments/assets/e4f35374-514f-4fdb-b344-6f62ee1afdb1)

"The grouping factor" mentioned above is a number of shared edges between clusters. We want to simplify as much as possible to reach 1/2 index count, therefore we want to lock as small amount of edges as possible. All these tips mean that we need to group those clusters that have the most shared edges because those edges will not be locked. Grouping is a common graph partitioning problem, where graph vertices (nodes) are clusters, edges are the fact that two clusters are connected and edge weights are the number of actual shared mesh edges between two clusters.

After splitting groups back, we add it to a cluster pool from which new clusters are generated. The pool is cleared after every simplification pass.

After all that, we will end up with a DAG (Directed Acyclic Graph) or basically a hierarchy of clusters, where the root of the graph is the lowest detail level with ideally 1 cluster. The next question is: "How do we traverse, select, and render those clusters?"

### The hierarchy
Due to the fact that merged groups are split back, we still cannot render all clusters independently - we can only render entire groups to avoid cluster overlap and/or missing clusters that will lead to holes in the mesh. It may sound bad, but groups are really small and their cluster count depends on an implementation and varies from 4 to 8. An average high-density mesh may have thousands of clusters at LOD 0, so we have ~1-3 thousand groups to render at LOD 0. So, if we LOD test cluster *X* and it fails (too detailed / too low detailed), we need to test all of its children groups and repeat it until we find a group that perfectly fits. What "fits" and what doesn't will be discussed below. You could also think that if we can only render entire groups, then we do not need to test all meshlets individually - we only need to test groups. The "selection" of groups form a *DAG cut*. See the photo below that shows what it looks like.

**Note**: the error **must** be in world-space in order to correctly LOD test groups.

![The DAG cut](https://github.com/user-attachments/assets/ed06c469-3371-43ed-9db9-43e0699bccc5)

### How to render
After mesh build, we end up with a hierarchy of clusters, where parents are simplified versions of their children. The traversal always starts from the root (least detailed LOD), assuming that most meshes will be rendered in low quality. During the build, each group "gets" a bounding sphere that encloses all vertices inside the group and the simplification error **and** its parent bounding sphere and error to allow entirely local (we only need group X data to test if it fits), parallel LOD selection - this is one of the most important parts.

### LOD test
We only render a group if it is detailed enough to not produce any perceptible error **and** its parent is simplified too much. The error is perceptible only if its projected sphere's radius is equal to or more than 1 pixel. If it is less than 1 pixel, then the error is imperceptible.

Now we need to understand what the "projected sphere" is. When testing a LOD, we form a sphere with a center at the group's bounding sphere and radius that equals the error, then we project it to the screen with current view parameters and get the size of the sphere in pixels - a metric used in LOD testing.

Important note: whenever a group with just enough error is found, we stop its subtree traversal because all of its children's LOD tests will fail due to the fact that the children and children's children will be too detailed to be rendered).

### Streaming
Now, when we have per-cluster LODs that form a hierarchy and understand how to render them, we get back to the question of memory. We can stream in and out *individual clusters*, keeping only needed resources and clusters resident. During the rendering, we request unloaded data on-demand. Then, the CPU loads needed resources and "lets GPU know" that resources are loaded so it use them in render.

This entire system is aimed at rendering meshes with the *higest quality possible* with high performance, so if a cluster is not resident, then we render its children that have higher quality to maintain graphics fidelity. 

## Conclusion
Let's sum up all the steps starting from mesh build to getting actual pixels on screen.

The mesh building process:
1. Split initial mesh to clusters
2. Add clusters to the pool
3. Group clusters based on connectivity
4. Merge each individual group's clusters into one index buffer
5. Simplify
6. Repeat until 1 cluster is left

After it is done, we have a hierarchy of clusters, where parents are simplified versions of their children.

The rendering process:
1. Start hierarchy traversal from the root
2. Project self group error sphere to the screen
3. Project parent group error sphere to the screen
4. Render if the self error is small enough and the parent error is too big
5. Stop subtree traversal

That's all the building blocks we need to build such a mesh rendering pipeline. In the next parts of this series, I will explain how to implement it from the very beginning to getting actual pixels on screen.

## Special thanks
This entire system is pretty much a Nanite rendering pipeline from Unreal Engine 5. Considering this fact, I would like to thank [Brian Karis](https://x.com/briankaris) and the entire Nanite and Unreal Engine team, without them it wouldn't be possible, at least for me, at least for now.

It is also worth mentioning the [SIGGRAPH "Deep Dive into Nanite talk by Brian Karis](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf), where this entire system (and more) is explained in details. This is the presentation I took images from.

And also, special thanks to [LVSTRI](https://github.com/LVSTRI) and [jglrxavpok](https://mastodon.gamedev.place/@jglrxavpok) for support and help during the development of this system.

Thanks for reading!
