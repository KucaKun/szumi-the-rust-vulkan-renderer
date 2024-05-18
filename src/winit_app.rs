use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::renderer::Renderer;

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
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
