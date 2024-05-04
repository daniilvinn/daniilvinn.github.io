# Vertex quantization in Omniforce Game Engine
## Renderer design considerations
From the very beginning of development of my engine's renderer, I wanted it to support highly detailed _meshes_.

To make that possible, my renderer required a vertex compression system which not just quantizes vertex positions to 16-bit floats. I needed higher compression ratio and very fast, near-instant decoding algorithm alongside with minimal precision loss.
Meshes in my engine's target scenes can have more than 200k polygons - which effectively proves my statement above about compression system requirements. To summarize, I can highlight these features which have to be present in quantization system:
- High compression rate, more than constant 50%
- Capability of runtime decoding
- Nearly instant decoding time
- Random access from Visibility Buffer

## Implementation
### Normal compression
For normal encoding, I chose good-old Octahedron-encoding method mentioned at [this page](https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/) which turned out to be the best option among others. It works fairly simple - normals are unit vectors, hence they basically represent points on unit sphere. Sphere can be divided into 8 "sections", effectively forming an octahedron, which then gets unfolded to a 2D plane, meaning that we can use `vec2` normals instead of `vec3`!

For further compression, I decided to compress `vec2` which was returned after Octahedron encoding to 16-bit float. It takes some good bit of precision, but considering that most of the materials use normal maps which are 8-bit - I decided that final precision loss is acceptable - it varied around 0.015. Using this method, I effectively compressed 12-bytes `vec3` normal to 4-bytes `fp16vec2` normal with acceptable precision loss.

#### Encoding (C++, using GLM library for math):
```cpp
glm::vec2 OctWrap(glm::vec2 v) {
	glm::vec2 w = 1.0f - glm::abs(glm::vec2(v.y, v.x));
	if (v.x < 0.0f) w.x = -w.x;
	if (v.y < 0.0f) w.y = -w.y;
	return w;
}

glm::u16vec2 QuantizeNormal(glm::vec3 n) {
	n /= (glm::abs(n.x) + glm::abs(n.y) + glm::abs(n.z));
	n = glm::vec3(n.z > 0.0f ? glm::vec2(n.x, n.y) : OctWrap(glm::vec2(n.x, n.y)), n.z);
	n = glm::vec3(glm::vec2(n.x, n.y) * 0.5f + 0.5f, n.z);

	return glm::packHalf(glm::vec2(n.x, n.y));
}
```
#### Decoding (GLSL)
```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

f16vec3 DecodeNormal(f16vec2 f)
{
	f = f * 2.0hf - 1.0hf;

	f16vec3 n = f16vec3(f.x, f.y, 1.0hf - abs(f.x) - abs(f.y));
	float16_t t = max(-n.z, 0.0hf);
	n.x += n.x >= 0.0hf ? -t : t;
	n.y += n.y >= 0.0hf ? -t : t;

	return normalize(n);
}
```

### Tangent compression
As well as normals, tangents have multiple ways to be quantized. One of the options was using "implicit tangents" - a single `float32` or `float16`, describing an angle around normal. With such method, decoding is done with _Rodrigues’ Rotation Formula_.

However, as I mentioned above - _very_ fast decoding time is one the biggest requirements. With the "implicit tangent" technique, I could desribe a tangent as angle encoded 16-bit float, but decoding would involve transcendental GPU operations, which I wanted to avoid. Also, considering that such method would only save 2 bytes per vertex compared to my implementation, I decided to go another way.

Tangents are regular unit vectors, as well as normal, which means that we can use the same encoding algorithm - Octahedron -> fp16! However, things get a little bit trickier when we also need to encode _bitangent sign_. Thankfully, sign is just 2 values, meaning that it can be represented using a single bit. With some 
