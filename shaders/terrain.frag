#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

out vec4 FragColor;

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform vec3 fogColor;
uniform float fogDensity;
uniform float fogStart;

void main()
{
    vec3 norm = normalize(Normal);
    
    // Strong ambient so terrain is always visible
    vec3 ambient = 0.4 * Color;
    
    // Diffuse (sun)
    vec3 lightDirection = normalize(lightDir);
    float diff = max(dot(norm, lightDirection), 0.0);
    vec3 diffuse = diff * Color * 0.6;
    
    // Sky light (from above) - subtle fill light
    float skyLight = max(dot(norm, vec3(0.0, 1.0, 0.0)), 0.0) * 0.15;
    vec3 sky = skyLight * vec3(0.8, 0.85, 1.0);
    
    // Combine - this is the solid terrain color
    vec3 result = ambient + diffuse + sky;
    
    // Optional fog (only if fogDensity > 0)
    if (fogDensity > 0.0) {
        float dist = length(viewPos - FragPos);
        float fogFactor = exp(-fogDensity * max(0.0, dist - fogStart));
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        result = mix(fogColor, result, fogFactor);
    }
    
    FragColor = vec4(result, 1.0);
}