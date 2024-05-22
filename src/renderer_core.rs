mod buffer_structs;
mod shaders;

use std::sync::Arc;

use std::ops::Deref;

use crate::vulkan_api_connection::VulkanConnection;
use nalgebra::Matrix4;
use nalgebra::Orthographic3;
use vulkano::buffer::Buffer;
use vulkano::buffer::BufferCreateInfo;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::command_buffer::SubpassEndInfo;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::Image;
use vulkano::image::ImageCreateInfo;
use vulkano::image::ImageType;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::color_blend::ColorBlendAttachmentState;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::render_pass::Framebuffer;
use vulkano::render_pass::FramebufferCreateInfo;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano::shader::EntryPoint;
use vulkano::shader::ShaderModule;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::SwapchainCreateInfo;

use self::buffer_structs::MyVertex;
use self::buffer_structs::MVP;

// Core is the struct that holds objects that depend on window size. They need to be remade each time a window is resized.
pub struct RendererCore {
    vapi: Arc<VulkanConnection>,
    images: Vec<Arc<Image>>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    pub command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub swapchain: Arc<Swapchain>,
    vertex_buffer: Arc<Subbuffer<[MyVertex]>>,
}
impl RendererCore {
    pub fn new(vapi: Arc<VulkanConnection>, dimensions: [u32; 2]) -> Self {
        let (swapchain, images) = RendererCore::create_swapchain(vapi.clone(), dimensions);

        let render_pass = RendererCore::get_render_pass(vapi.device.clone(), swapchain.clone());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vapi.device.clone()));

        let framebuffers = RendererCore::get_framebuffers(&images, &render_pass);

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            vapi.device.clone(),
            Default::default(),
        ));

        let (vs, fs) = RendererCore::get_shaders(vapi.device.clone());

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [1024.0, 1024.0],
            depth_range: 0.0..=1.0,
        };

        let pipeline = RendererCore::get_pipeline(
            vapi.device.clone(),
            vs.entry_point("main").unwrap(),
            fs.entry_point("main").unwrap(),
            render_pass.clone(),
            viewport.clone(),
        );

        let vertex_buffer = Arc::new(RendererCore::get_triangle_vertex_buffer(
            memory_allocator.clone(),
            vec![
                MyVertex {
                    position: [100, 100],
                    color: [255, 0, 35],
                },
                MyVertex {
                    position: [200, 100],
                    color: [0, 255, 50],
                },
                MyVertex {
                    position: [150, 200],
                    color: [0, 100, 255],
                },
            ],
        ));
        let mvp_buffer = Arc::new(RendererCore::get_mvp_buffer(
            memory_allocator.clone(),
            viewport.clone(),
        ));
        let mvp_set = RendererCore::get_mvp_descriptor_set(
            vapi.device.clone(),
            pipeline.clone(),
            mvp_buffer.clone(),
        );
        let command_buffers = RendererCore::get_command_buffers(
            &command_buffer_allocator,
            &vapi.queue,
            &pipeline,
            &framebuffers,
            &vertex_buffer,
            vec![mvp_set],
        );
        Self {
            vapi,
            viewport,
            images,
            framebuffers,
            command_buffers,
            render_pass,
            swapchain,
            memory_allocator,
            command_buffer_allocator,
            vertex_buffer,
            pipeline,
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
        self.framebuffers = RendererCore::get_framebuffers(&self.images, &self.render_pass);
        self.viewport.extent = [dimensions[0] as f32, dimensions[1] as f32];
        let (vs, fs) = RendererCore::get_shaders(self.vapi.device.clone());
        let pipeline = RendererCore::get_pipeline(
            self.vapi.device.clone(),
            vs.entry_point("main").unwrap(),
            fs.entry_point("main").unwrap(),
            self.render_pass.clone(),
            self.viewport.clone(),
        );
        let mvp_buffer = Arc::new(RendererCore::get_mvp_buffer(
            self.memory_allocator.clone(),
            self.viewport.clone(),
        ));
        let mvp_set = RendererCore::get_mvp_descriptor_set(
            self.vapi.device.clone(),
            pipeline.clone(),
            mvp_buffer.clone(),
        );
        self.command_buffers = RendererCore::get_command_buffers(
            &self.command_buffer_allocator,
            &self.vapi.queue,
            &pipeline,
            &self.framebuffers,
            &self.vertex_buffer,
            vec![mvp_set],
        );
    }

    fn create_swapchain(
        vapi: Arc<VulkanConnection>,
        dimensions: [u32; 2],
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let composite_alpha = vapi
            .surface_caps
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();

        let image_format = vapi
            .physical_device
            .surface_formats(&vapi.surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            vapi.device.clone(),
            vapi.surface.clone(),
            SwapchainCreateInfo {
                min_image_count: vapi.surface_caps.min_image_count + 1, // How many buffers to use in the swapchain
                image_format,
                image_extent: dimensions,
                image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();
        return (swapchain, images);
    }

    fn create_image_view(self, dimensions: [u32; 3]) -> Arc<ImageView> {
        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: dimensions,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        ImageView::new_default(image.clone()).unwrap()
    }

    fn get_mvp_buffer(
        memory_allocator: Arc<
            vulkano::memory::allocator::GenericMemoryAllocator<
                vulkano::memory::allocator::FreeListAllocator,
            >,
        >,
        viewport: Viewport,
    ) -> Subbuffer<MVP> {
        let model: Matrix4<f32> = Matrix4::identity();
        let view: Matrix4<f32> = Matrix4::identity();
        let projection =
            Orthographic3::new(0.0, viewport.extent[1], 0.0, viewport.extent[1], -1.0, 1.0)
                .to_homogeneous();
        let mvp = MVP {
            model: model.into(),
            view: view.into(),
            proj: projection.into(),
        };
        let uniform_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mvp,
        )
        .unwrap();
        uniform_buffer
    }

    fn get_triangle_vertex_buffer(
        memory_allocator: Arc<
            vulkano::memory::allocator::GenericMemoryAllocator<
                vulkano::memory::allocator::FreeListAllocator,
            >,
        >,
        points: Vec<MyVertex>,
    ) -> vulkano::buffer::Subbuffer<[MyVertex]> {
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            points,
        )
        .unwrap();
        vertex_buffer
    }

    fn get_command_buffers(
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        pipeline: &Arc<GraphicsPipeline>,
        framebuffers: &Vec<Arc<Framebuffer>>,
        vertex_buffer: &Subbuffer<[MyVertex]>,
        descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        framebuffers
            .iter()
            .map(|framebuffer| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator,
                    queue.queue_family_index(),
                    // Don't forget to write the correct buffer usage.
                    CommandBufferUsage::MultipleSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        pipeline.bind_point(),
                        pipeline.layout().clone(),
                        0,
                        descriptor_sets.clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap();

                builder.build().unwrap()
            })
            .collect()
    }

    fn get_mvp_descriptor_set(
        device: Arc<Device>,
        pipeline: Arc<GraphicsPipeline>,
        buffer: Arc<Subbuffer<MVP>>,
    ) -> Arc<PersistentDescriptorSet> {
        let descriptor_set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());
        let descriptor_writes = [WriteDescriptorSet::buffer(0, buffer.deref().clone())];
        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout,
            descriptor_writes,
            [],
        )
        .unwrap();
        descriptor_set
    }

    fn get_pipeline(
        device: Arc<Device>,
        vs_entry_point: EntryPoint,
        fs_entry_point: EntryPoint,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) -> Arc<GraphicsPipeline> {
        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs_entry_point.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry_point),
            PipelineShaderStageCreateInfo::new(fs_entry_point),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }

    fn get_shaders(device: Arc<Device>) -> (Arc<ShaderModule>, Arc<ShaderModule>) {
        (
            shaders::vs::load(device.clone()).expect("failed to create shader module"),
            shaders::fs::load(device.clone()).expect("failed to create shader module"),
        )
    }

    fn get_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            device,
            attachments: {
                color: {
                    // Set the format the same as the swapchain.
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap()
    }
}
