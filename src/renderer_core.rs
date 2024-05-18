use std::sync::Arc;

use vulkano::{
    buffer::Subbuffer,
    command_buffer::{allocator::StandardCommandBufferAllocator, PrimaryAutoCommandBuffer},
    image::{view::ImageView, Image},
    memory::allocator::StandardMemoryAllocator,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, RenderPass},
    swapchain::{Swapchain, SwapchainCreateInfo},
};

use crate::{
    prepare::{self, MyVertex},
    vulkan_api_connection::VulkanConnection,
};

// Core is the struct that holds objects that depend on window size. They need to be remade each time a window is resized.
pub struct RendererCore {
    vapi: Arc<VulkanConnection>,
    images: Vec<Arc<Image>>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    view: Arc<ImageView>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pub command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub swapchain: Arc<Swapchain>,
    vertex_buffer: Arc<Subbuffer<[MyVertex]>>,
}
impl RendererCore {
    pub fn new(vapi: Arc<VulkanConnection>, dimensions: [u32; 2]) -> Self {
        let (swapchain, images) = prepare::create_swapchain(
            vapi.physical_device.clone(),
            vapi.device.clone(),
            vapi.surface.clone(),
            vapi.surface_caps.clone(),
            dimensions,
        );

        let render_pass = prepare::get_render_pass(vapi.device.clone(), swapchain.clone());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vapi.device.clone()));

        let framebuffers = prepare::get_framebuffers(&images, &render_pass);

        let view = prepare::create_image_view(memory_allocator.clone(), [1024, 1024, 1]);

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            vapi.device.clone(),
            Default::default(),
        ));

        let (vs, fs) = prepare::get_shaders(vapi.device.clone());

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [1024.0, 1024.0],
            depth_range: 0.0..=1.0,
        };

        let pipeline = prepare::get_pipeline(
            vapi.device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );

        let vertex_buffer = Arc::new(prepare::get_triangle_vertex_buffer(
            memory_allocator.clone(),
        ));

        let command_buffers = prepare::get_command_buffers(
            &command_buffer_allocator,
            &vapi.queue,
            &pipeline,
            &framebuffers,
            &vertex_buffer,
        );
        Self {
            vapi,
            viewport,
            images,
            framebuffers,
            command_buffers,
            render_pass,
            swapchain,
            view,
            memory_allocator,
            command_buffer_allocator,
            vertex_buffer,
        }
    }

    pub fn recreate(&mut self, dimensions: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: dimensions,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain: {e}");
        self.swapchain = new_swapchain;
        self.images = new_images;
        self.framebuffers = prepare::get_framebuffers(&self.images, &self.render_pass);
        self.viewport.extent = [dimensions[0] as f32, dimensions[1] as f32];
        let (vs, fs) = prepare::get_shaders(self.vapi.device.clone());
        let pipeline = prepare::get_pipeline(
            self.vapi.device.clone(),
            vs.clone(),
            fs.clone(),
            self.render_pass.clone(),
            self.viewport.clone(),
        );

        self.command_buffers = prepare::get_command_buffers(
            &self.command_buffer_allocator,
            &self.vapi.queue,
            &pipeline,
            &self.framebuffers,
            &self.vertex_buffer,
        );
    }
}
