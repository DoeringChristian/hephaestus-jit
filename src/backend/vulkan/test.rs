use crate::backend;

#[test]
fn image() {
    let device = backend::Device::vulkan(0).unwrap();

    let tex = device.create_texture([100, 100, 100], 4).unwrap();
}
