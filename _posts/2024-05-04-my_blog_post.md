## Vertex quantization in Omniforce Game Engine
### Renderer design considerations
  From the very beginning of development of my engine's renderer, I wanted it to support highly detailed _meshes_. 
To make that possible, my renderer required a vertex compression system which not just quantizes vertex positions to 16-bit floats. I needed higher compression ratio and very fast, near-instant decoding algorithm alongside with minimal precision loss.
  Meshes in my engine's target scenes can have more than 200k polygons - which effectively proves my statement above about compression system requirements. To summarize, I can highlight these features which have to be present in quantization system:
- High compression rate (more than constant 50% rate)
- Capability of runtime decoding
- Nearly instant decoding speed
- Random access from Visibility Buffer
