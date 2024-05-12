use std::sync::Arc;
use vulkano::memory::allocator::StandardMemoryAllocator;
mod mandelbrot;
mod prepare;

fn main() {
    let (device, queue) = prepare::prepare_device();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    mandelbrot::mandelbrot(device, queue, memory_allocator);
}
