shader_type canvas_item;

uniform ivec2 ScreenResolution = ivec2(1920, 1080);
uniform sampler2D ScreenTexture : hint_screen_texture;

uniform sampler2D Map;

void fragment() {
	ivec2 raw_coordinate = ivec2(FRAGCOORD.xy);
	
	vec2 _uv = vec2(0.0);
	uvec4 data = uvec4(texelFetch(Map, raw_coordinate, 0) * 255.0);
	 _uv.x = uintBitsToFloat(
		data.r |
		(data.g << 8u) |
		(data.b << 16u) |
		(data.a << 24u)
	);
	
	raw_coordinate.x += ScreenResolution.x;
	data = uvec4(texelFetch(Map, raw_coordinate, 0) * 255.0);
	_uv.y = uintBitsToFloat(
		data.r |
		(data.g << 8u) |
		(data.b << 16u) |
		(data.a << 24u)
	);
	
	COLOR = any(lessThan(_uv, vec2(0.0))) || any(greaterThan(_uv, vec2(1.0))) ?
		vec4(0.0, 0.0, 0.0, 1.0) :
		texture(ScreenTexture, _uv);
}
