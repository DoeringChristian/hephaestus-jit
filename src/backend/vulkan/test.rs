use super::presenter::{Presenter, PresenterInfo};
use super::*;
use crate::backend;

#[test]
fn image() {
    let device = backend::Device::vulkan(0).unwrap();

    let tex = device.create_texture([100, 100, 100], 4).unwrap();

    dbg!(&tex);
}
#[test]
fn presenter() {
    let device = Device::create(0);

    let info = PresenterInfo {
        width: 800,
        height: 600,
    };
    let presenter = Presenter::create(&device, &info);

    loop {
        presenter.present_image();
    }
}
