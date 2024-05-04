## Vertex quantization in Omniforce Game Engine
### Renderer design considerations
From the very beginning of development of my engine's renderer, I wanted it to support highly detailed _meshes_. 
To make that possible, I need a vertex compression system which not just quantizes vertex positions to fp16. I needed bigger compression ratio. 
