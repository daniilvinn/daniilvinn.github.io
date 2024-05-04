# Vertex quantization in Omniforce Game Engine
## Renderer design considerations
From the very beginning of development of my engine's renderer, I wanted it to support highly detailed _meshes_. To make that possible, my renderer required a vertex compression system which not just quantizes vertex positions to 16-bit floats. I needed higher compression ratio and very fast, near-instant decoding algorithm alongside with minimal precision loss.

Another thing to consider is that my engine's renderer is built around mesh shaders, meaning that all meshes are split into clusters with up to 64 vertices and 124 triangles. This is very important note, which will be used for position compression. Mesh shaders are supported in NVIDIA Turing+ and AMD RDNA2+ GPUs. On NVIDIA, FP16 and FP32 computing rates are the same starting from Ampere generation, however, I can benefit from it on Turing GPUs, where FP16 computing rate is doubled compared to FP32 - this is one more thing to consider when designing compression system.

Meshes in my engine's target scenes can have more than 200k polygons - which effectively proves my statement above about compression system requirements. 

To summarize, I can highlight these features which have to be present in quantization system:
- High compression rate, more than constant 50%
- Capability of runtime decoding
- Nearly instant decoding time
- Random access from Visibility Buffer

## Implementation, part 1. Attribute compression
### Normal compression
For normal encoding, I chose good-old Octahedron-encoding method mentioned at [this page](https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/) which turned out to be the best option among others. It works fairly simple - normals are unit vectors, hence they represent points on unit sphere. Sphere can be divided into 8 "sections", effectively forming an octahedron, which then gets unfolded to a 2D plane, meaning that we can use `vec2` normals instead of `vec3`!

For further compression, I decided to compress `vec2` which was returned after Octahedron encoding to 16-bit float. It takes some good bit of precision, but considering that most of the materials use normal maps which are 8-bit - I decided that final precision loss is acceptable - it varied around 0.015. Using this method, I effectively compressed 12-bytes `vec3` normal to 4-bytes `fp16vec2` normal with acceptable precision loss.

To summarize:
- Use octahedron encoding with further compression to fp16
- Precision loss is approximately 0.015
- Decoding kept simple, nearly free

#### Encoding code (C++, using GLM library for math):
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
#### Decoding code (GLSL)
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
![Visualization of octahedron encoding](https://www.jeremyong.com/images/diamond/octahedral.png)

### Tangent compression
As well as normals, tangents have multiple ways to be quantized. One of the options was using "implicit tangents" - a single `float32` or `float16`, describing an angle around normal. With such method, decoding is done with _Rodrigues’ Rotation Formula_.

However, as I mentioned above - _very_ fast decoding time is one the biggest requirements. With the "implicit tangent" technique, I could desribe a tangent as an angle, encoded in 16-bit float, but decoding would involve transcendental GPU operations (sin/cos), which I wanted to avoid as much as possible. Also, considering that such method would only save 2 bytes per vertex compared to my implementation, I decided to go another way.

Tangents are regular unit vectors, as well as normal, which means that we can use the same encoding algorithm - Octahedron -> fp16! However, things get a little bit trickier when we also need to encode _bitangent sign_. Thankfully, sign is just 2 values, meaning that it can be represented using a single bit. After little research, I came up with a solution - encode that single bit into octahedron-encoded tangent. **Very important note**: I encode bitangent sign bit *after* octahedron and fp16 compression, because if I do it vice-versa, the sign bit can be lost due to compression. I chose lowest bit of Y component - according to IEEE-754 float16 standard, it is lowest bit of mantissa, meaning that it has the least influence on final precision. After my tests, I ended up with approximately 0.03 precision loss for tangents.

Decoding is fairly simple - just use the same method of decoding as we used for normals with one little difference - extract sign bit before decoding using this snippet of code: `float16 sign = float16BitsToUint16(tangent.y) & 1us ? 1.0hf : -1.0hf`

To summarize:
- Tangents are octahedron-encoded with further compression to fp16
- Bitangent sign bit is copied to Y component's lowest bit
- Precision loss is approximately 0.03
- Decoding is as simple as normal decoding with extra step to extract sign bit

#### Encoding code (C++)
```cpp
glm::u16vec2 QuantizeTangent(glm::vec4 t) {
	glm::u16vec2 q = QuantizeNormal(glm::vec3(t.x, t.y, t.z));

	// If bitangent sign is positive, we set first bit of Y component to 1, otherwise (if it is negative) we set 0
	q.y = t.w == 1.0f ? q.y | 1u : q.y & ~1u;

	return q;
}
```

#### Decoding code (GLSL)
```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

f16vec4 DecodeTangent(f16vec2 f)
{
	f16vec4 t;
	t.xyz = DecodeNormal(f);
	
	// Check if lowest bit of Y's mantissa is set. If so, bitangent sign is positive, otherwise negative.
	t.w = bool(float16BitsToUint16(f.y) & 1us) ? 1.0hf : -1.0hf;

	return t;
}
```

### UV compression
Texture coordinates are simply compressed to fp16. However, it may lead to some problems with texture sampling when used on meshes with tiled textures - for example, a terrain.

## Implementation, part 2. Vertex position compression
Compared to attributes, vertex compression has much more space for creativity. When I was looking through "Deep dive into Nanite Virtualized Geometry" by Brian Karis, I was inspired by their vertex compression system, which is where I got the idea for my own implementation. 

Why not naive fp16 compression? The answer is simple - it doesn't suit _almost any_ of my requirements: vertex size is still comparably big (16 bits per channel) and it has large error with big meshes. I needed another, better way for vertex position compression.

When I was implementing my system, I kept in mind three factors:
- My renderer is cluster-based
- Float32 and float16 can represent values far beyond mesh AABB, which is reduntant and only introduces additional "worthless" bits
- Do I really need byte aligned data-structures? (spoiler: no)

### Grid quantization
#### Overview
Vertex quantization is done using **uniform grid**, where each vertex gets snapped to it. Grid step is an option - it can vary from 4 to 16 bits, which means that grid step varies from `1 / pow(2, 4)` to `1 / pow(2, 16)`. Compression is done with this code: `round(x * pow(2, grid_precision))`, which returns us a signed integer representing grid step count (essentially, a compressed value).
#### Avoiding geometry cracks
Considering that my renderer is cluster-based and is using mesh shading technology, my mesh is split up into "meshlets". Compressing them naively may introduce *cracks* between them (see a photo below), which is unacceptable. To solve this, we need to quantize all vertices, lods (if using meshlet-level lods) and all spatial data in general, including meshlets (see below), against *the same grid*.
#### Precision and bit size
Using this method, precision is kept very high. Maximum error equals to grid step size divided by 2, due to rounding. To calculate final bit size, I use this (preudo-) code: `ceil(log2(round(f * pow(2, precision)))`. For example, if I want to get final bit size of 1D vertex at 5.0 quantized against 8-bit grid, I do this: `ceil(log2(round(5.0 * pow(2, 8)))`, which returns 11.
![Geometry cracks](https://files.fm/u/3efu72jjg9)

### Encoding in meshlet-space
To compress vertices even further, I encode them in meshlet-space, instead of local space. It can save us good 1-6 bits (3-5 in average) per vertex channel, depending on meshlet spatial size. Now, this is how I encode vertex:
`round((f - meshlet-center) * pow(2, precision))`.

We can calculate new vertex bit size using this technique. Let's say, we still have 1D vertex laying at 5.0 and it belongs to a meshlet with center at 4.2. Now, to calculate new bit size, we use this: `ceil(log2(round((5.0 - 4.2) * pow(2, 8)))`, and it returns us 8 - which is 3 bits less than previous result, meaning that we saved additional 3 bits **per channel**.

Important note: meshlet centers **must** be quantized as well in order to avoid geometry cracks between meshlets.

### Bit stream
All this compression and 3-5 bits savings don't make any sense if I still have byte-aligned data structures. I needed a data structure which is not aligned to a byte - I needed a bit stream.

Using a bit stream instead of array of float32/float16/uint/int, I can compress vertices much effectively, because not a single bit will be wasted: data is packed as much as possible. This technique turned out to work really well on GPU - I simply use GPU bit stream reader to fetch vertex data. By providing an offset in *bits* and num of bits to read, the reader returns us a `uint` representing encoded value.

Considering that I deal with non-byte-aligned data, I can't just load some value from bit stream and expect it to have sign. To solve this, I used a trick called "bit extend", which expands sign bit, effectively restoring sign of the value.
