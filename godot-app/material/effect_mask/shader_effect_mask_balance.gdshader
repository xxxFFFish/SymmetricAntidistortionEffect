shader_type canvas_item;

uniform ivec2 ScreenResolution = ivec2(1920, 1080);
uniform sampler2D ScreenTexture : hint_screen_texture;

uniform sampler2D Map;

void fragment() {
	ivec2 raw_coordinate = ivec2(FRAGCOORD.xy);
	
	ivec2 coef = (2 * raw_coordinate) / ScreenResolution;
	ivec2 map_coordinate = (ivec2(1) - 2 * coef) * raw_coordinate +
		coef * (ScreenResolution - ivec2(1));
	
	vec2 _uv = vec2(0.0);
	uvec4 data = uvec4(texelFetch(Map, map_coordinate, 0) * 255.0);
	 _uv.x = uintBitsToFloat(
		data.r |
		(data.g << 8u) |
		(data.b << 16u) |
		(data.a << 24u)
	);
	
	map_coordinate.x += ScreenResolution.x / 2;
	data = uvec4(texelFetch(Map, map_coordinate, 0) * 255.0);
	_uv.y = uintBitsToFloat(
		data.r |
		(data.g << 8u) |
		(data.b << 16u) |
		(data.a << 24u)
	);
	
	vec2 f_coef = vec2(coef);
	_uv = (vec2(1.0) - 2.0 * f_coef) * _uv + f_coef;
	
	COLOR = any(lessThan(_uv, vec2(0.0))) || any(greaterThan(_uv, vec2(1.0))) ?
		vec4(0.0, 0.0, 0.0, 1.0) :
		texture(ScreenTexture, _uv);
}
