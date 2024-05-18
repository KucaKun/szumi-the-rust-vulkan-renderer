use std::sync::Arc;

use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    swapchain::{Surface, SurfaceCapabilities},
    VulkanLibrary,
};
use winit::window::Window;

use crate::prepare;

/// This struct does not change during the lifetime of the application
pub struct VulkanConnection {
    pub device: Arc<Device>,
    pub physical_device: Arc<PhysicalDevice>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub surface_caps: SurfaceCapabilities,
}
impl VulkanConnection {
    pub fn new(window: Arc<Window>) -> Self {
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
