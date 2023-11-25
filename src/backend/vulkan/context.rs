use std::ops::Deref;

use super::buffer;
use super::Device;
use ash::vk;

pub struct Context {
    pub device: Device,
    pub cb: vk::CommandBuffer,
    pub buffers: Vec<buffer::Buffer>,
    pub image_views: Vec<vk::ImageView>,
}

impl Deref for Context {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Context {
    pub fn new(device: &Device, cb: vk::CommandBuffer) -> Self {
        Self {
            device: device.clone(),
            cb,
            buffers: vec![],
            image_views: vec![],
        }
    }
    pub fn buffer(&mut self, info: buffer::BufferInfo) -> &buffer::Buffer {
        self.buffers
            .push(buffer::Buffer::create(&self.device, info));
        self.buffers.last().unwrap()
    }
    pub fn buffer_mut(&mut self, info: buffer::BufferInfo) -> &mut buffer::Buffer {
        self.buffers
            .push(buffer::Buffer::create(&self.device, info));
        self.buffers.last_mut().unwrap()
    }
    pub fn create_image_view(&mut self, info: &vk::ImageViewCreateInfo) -> vk::ImageView {
        unsafe { self.device.create_image_view(info, None).unwrap() }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view, None);
            }
        }
    }
}
