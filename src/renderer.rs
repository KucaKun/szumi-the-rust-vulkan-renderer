use std::sync::Arc;

use vulkano::{
    swapchain::{self, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated,
};
use winit::window::Window;

use crate::{renderer_core::RendererCore, vulkan_api_connection::VulkanConnection};

pub struct Renderer {
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
