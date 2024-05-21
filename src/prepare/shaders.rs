pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
                #version 460
    
                layout(location = 0) in ivec2 position;
                layout(location = 1) in uvec3 color;

                layout(location = 0) out vec3 v_color;

                layout(binding = 0) uniform UniformBufferObject {
                    mat4 model;
                    mat4 view;
                    mat4 proj;
                } mvp;

                void main() {
                    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(position, 0.0, 1.0);
                    v_color = color/255.0;
                }
            ",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
                #version 460
    
                layout(location = 0) out vec4 f_color;

                layout(location = 0) in vec3 v_color;                

                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            ",
    }
}
