use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub(crate) struct MyVertex {
    #[format(R32G32_SINT)]
    pub position: [i32; 2],

    #[format(R8G8B8_UINT)]
    pub color: [u8; 3],
}

#[derive(BufferContents)]
#[repr(C)]
pub(crate) struct MVP {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}
