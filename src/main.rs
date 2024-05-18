mod prepare;
mod renderer;
mod renderer_core;
mod vulkan_api_connection;
mod winit_app;

use winit::event_loop::{ControlFlow, EventLoop};
use winit_app::App;

fn main() {
    let mut app: App = App::default();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).expect("Error on running app");
}
