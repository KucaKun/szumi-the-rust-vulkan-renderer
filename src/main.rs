use std::sync::Arc;

use prepare::MyVertex;
use vulkano::buffer::Subbuffer;
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
use vulkano::render_pass::{self, Framebuffer, RenderPass};
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
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
}
/// This struct does not change during the lifetime of the application
struct VulkanConnection {
    device: Arc<Device>,
    physical_device: Arc<PhysicalDevice>,
    queue: Arc<Queue>,
    surface: Arc<Surface>,
    surface_caps: SurfaceCapabilities,
}
impl VulkanConnection {
    fn new(window: Arc<Window>) -> Self {
        let instance_extensions = Surface::required_extensions(window.clone().as_ref());
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: instance_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
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

        let surface_caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        Self {
            device,
            physical_device,
            queue: queues.next().unwrap(),
            surface,
            surface_caps,
        }
    }
}
// Core is the struct that holds objects that depend on window size. They need to be remade each time a window is resized.
struct RendererCore {
    vapi: Arc<VulkanConnection>,
    images: Vec<Arc<Image>>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    view: Arc<ImageView>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    swapchain: Arc<Swapchain>,
    vertex_buffer: Arc<Subbuffer<[MyVertex]>>,
}

impl RendererCore {
    fn new(vapi: Arc<VulkanConnection>, dimensions: [u32; 2]) -> Self {
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

    fn recreate(&mut self, dimensions: [u32; 2]) {
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
struct Renderer {
    vapi: Arc<VulkanConnection>,
    core: RendererCore,
    last_frame_future: Option<Box<dyn GpuFuture>>,
}
impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let vapi = Arc::new(VulkanConnection::new(window.clone()));
        let core = RendererCore::new(vapi.clone(), [1024, 1024]);
        Self {
            vapi,
            core,
            last_frame_future: None,
        }
    }

    /// This method recreates everything that depends on the window size
    pub fn recreate_core(&mut self, window: Arc<Window>) {
        let dimensions = window.inner_size().into();
        self.core.recreate(dimensions);
    }

    pub fn on_draw(&mut self, window: Arc<Window>) {
        // Acquire the next image to render to
        let (image_i, _suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.core.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if _suboptimal {
            self.recreate_core(window.clone());
            return;
        }

        // Execute the command buffer
        let execution = sync::now(self.vapi.device.clone())
            .join(acquire_future)
            .then_execute(
                self.vapi.queue.clone(),
                self.core.command_buffers[image_i as usize].clone(),
            )
            .unwrap()
            .then_swapchain_present(
                self.vapi.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.core.swapchain.clone(), image_i),
            )
            .then_signal_fence_and_flush();

        match execution.map_err(Validated::unwrap) {
            Ok(future) => {
                // Two frames in flight
                if self.last_frame_future.is_some() {
                    self.last_frame_future.as_mut().unwrap().cleanup_finished();
                }
                self.last_frame_future = Some(Box::new(future));
            }
            Err(e) => {
                println!("failed to flush future: {e}");
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan Triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 1024.0));
        self.window = Some(Arc::new(
            event_loop.create_window(window_attributes).unwrap(),
        ));
        self.renderer = Some(Renderer::new(
            self.window
                .as_ref()
                .expect("Window should be set before renderer")
                .clone(),
        ));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        assert!(self.window.is_some());
        assert!(self.renderer.is_some());
        let window = self.window.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        //MARK: - Event loop
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                println!("The window was resized to {:?}", new_size);
                renderer.recreate_core(window.clone());
            }
            WindowEvent::RedrawRequested => {
                renderer.on_draw(window.clone());
                window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let mut app: App = App::default();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).expect("Error on running app");
}
