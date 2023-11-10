use crate::{backend, tr};

#[test]
fn simple() {
    let device = backend::Device::vulkan(0).unwrap();

    let i = tr::index(10);
    let j = tr::index(5);

    j.add(&tr::literal(1u32)).scatter(&i, &j);

    j.schedule();

    let graph = tr::compile();
    graph.launch_slow(&device);

    dbg!(graph.n_passes());

    dbg!(&i.data().buffer().unwrap().to_host::<u32>());
    dbg!(&j.data().buffer().unwrap().to_host::<u32>());
    assert_eq!(
        i.data().buffer().unwrap().to_host::<u32>().unwrap(),
        vec![1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    );
    assert_eq!(
        j.data().buffer().unwrap().to_host::<u32>().unwrap(),
        vec![0, 1, 2, 3, 4]
    );
}

#[test]
fn simple_u16() {
    let device = backend::Device::vulkan(0).unwrap();

    let c = tr::sized_literal(1u16, 10);
    c.schedule();

    tr::compile().launch_slow(&device);

    dbg!(&c.data().buffer().unwrap().to_host::<u16>().unwrap());
}

#[test]
fn scatter_chain() {
    let device = backend::Device::vulkan(0).unwrap();

    let b0 = tr::sized_literal(0, 5);

    tr::literal(1).scatter(&b0, &tr::index(10));

    let b1 = b0.add(&tr::literal(1));
    b1.schedule();

    tr::compile().launch_slow(&device);

    dbg!(&b1.data().buffer().unwrap().to_host::<i32>().unwrap());
}
