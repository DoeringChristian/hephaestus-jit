use self::backend::Device;

mod backend;
mod trace;
mod tracer;

fn main() {
    let device = Device::cuda(0).unwrap();
    let buffer = device.create_buffer(10).unwrap();
    dbg!(buffer);
    dbg!(device);
}
