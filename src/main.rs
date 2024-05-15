use std::sync::Arc;

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::swapchain::{
    self, Surface, SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Validated, VulkanError, VulkanLibrary};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::Window;
use winit::window::WindowId;

mod prepare;

#[derive(Default)]
struct App {
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    device: Option<Arc<Device>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    images: Vec<Arc<Image>>,
    physical_device: Option<Arc<PhysicalDevice>>,
    queue: Option<Arc<Queue>>,
    render_pass: Option<Arc<RenderPass>>,
    surface: Option<Arc<Surface>>,
    surface_caps: Option<SurfaceCapabilities>,
    swapchain: Option<Arc<Swapchain>>,
    view: Option<Arc<ImageView>>,
    viewport: Viewport,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        ));
        let window = self.window.as_ref().unwrap();

        let required_extensions = Surface::required_extensions(window);

        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        // Device and Queue
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::default()
        };

        let (physical_device, queue_family_index) =
            prepare::select_physical_device(&instance, &surface, &device_extensions);

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
        self.device = Some(device);
        self.queue = Some(queues.next().unwrap());

        self.surface_caps = Some(
            physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities"),
        );
        self.physical_device = Some(physical_device);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        //MARK: - Event loop
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                println!("The window was resized to {:?}", new_size);

                let window = self.window.as_ref().unwrap();
                let device = self.device.as_ref().unwrap();
                let physical_device = self.physical_device.as_ref().unwrap();
                let surface = self.surface.as_ref().unwrap();
                let queue = self.queue.as_ref().unwrap();
                let surface_caps = self.surface_caps.as_ref().unwrap();

                // Swapchain
                let dimensions = window.inner_size();
                let composite_alpha = surface_caps
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap();
                let image_format = physical_device
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0;

                let (swapchain, images) = Swapchain::new(
                    device.clone(),
                    surface.clone(),
                    SwapchainCreateInfo {
                        min_image_count: surface_caps.min_image_count + 1, // How many buffers to use in the swapchain
                        image_format,
                        image_extent: dimensions.into(),
                        image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
                        composite_alpha,
                        ..Default::default()
                    },
                )
                .unwrap();

                // Render Pass
                let render_pass = prepare::get_render_pass(device.clone(), &swapchain);
                let framebuffers = prepare::get_framebuffers(&images, &render_pass);
                let memory_allocator =
                    Arc::new(StandardMemoryAllocator::new_default(device.clone()));
                let image = Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8G8B8A8_UNORM,
                        extent: [1024, 1024, 1],
                        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )
                .unwrap();

                let vertex_buffer = prepare::get_triangle_vertex_buffer(memory_allocator);

                let viewport = Viewport {
                    offset: [0.0, 0.0],
                    extent: new_size.into(),
                    depth_range: 0.0..=1.0,
                };

                let (vs, fs) = prepare::get_shaders(device.clone());

                let pipeline = prepare::get_pipeline(
                    device.clone(),
                    vs.clone(),
                    fs.clone(),
                    render_pass.clone(),
                    viewport.clone(),
                );

                let command_buffer_allocator =
                    StandardCommandBufferAllocator::new(device.clone(), Default::default());

                self.command_buffers = prepare::get_command_buffers(
                    &command_buffer_allocator,
                    &queue,
                    &pipeline,
                    &framebuffers,
                    &vertex_buffer,
                );

                self.view = Some(ImageView::new_default(image.clone()).unwrap());
                self.swapchain = Some(swapchain);
                self.images = images;
                self.framebuffers = framebuffers;
                self.viewport = viewport;
                self.render_pass = Some(render_pass);
            }
            WindowEvent::RedrawRequested => {
                let window = self.window.as_ref().unwrap();
                let device = self.device.as_ref().unwrap();
                let physical_device = self.physical_device.as_ref().unwrap();
                let surface = self.surface.as_ref().unwrap();
                let queue = self.queue.as_ref().unwrap();
                let surface_caps = self.surface_caps.as_ref().unwrap();
                let swapchain = self.swapchain.as_ref().unwrap();

                let (image_i, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                let execution = sync::now(device.clone())
                    .join(acquire_future)
                    .then_execute(
                        queue.clone(),
                        self.command_buffers[image_i as usize].clone(),
                    )
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                    )
                    .then_signal_fence_and_flush();

                match execution.map_err(Validated::unwrap) {
                    Ok(future) => {
                        // Wait for the GPU to finish.
                        future.wait(None).unwrap();
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                    }
                }

                window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let mut app = App::default();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).expect("Error on running app");
}
