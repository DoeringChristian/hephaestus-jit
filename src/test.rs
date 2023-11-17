use crate::{backend, tr, vulkan};

#[test]
fn simple() {
    let device = backend::Device::vulkan(0).unwrap();

    let i = tr::index(10);
    let j = tr::index(5);

    j.add(&tr::literal(1u32)).scatter(&i, &j);

    j.schedule();

    let graph = tr::compile();
    graph.launch(&device);

    dbg!(graph.n_passes());

    dbg!(i.to_vec::<u32>());
    dbg!(j.to_vec::<u32>());
    assert_eq!(i.to_vec::<u32>(), vec![1, 2, 3, 4, 5, 5, 6, 7, 8, 9]);
    assert_eq!(j.to_vec::<u32>(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn simple_u16() {
    let device = backend::Device::vulkan(0).unwrap();

    let c = tr::sized_literal(1u16, 10);
    c.schedule();

    tr::compile().launch(&device);

    dbg!(c.to_vec::<u16>());
    assert_eq!(c.to_vec::<u16>(), vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
}

#[test]
fn scatter_chain() {
    let device = backend::Device::vulkan(0).unwrap();

    let b0 = tr::sized_literal(0, 5);

    tr::literal(1).scatter_if(&b0, &tr::index(10), &tr::literal(true));

    let b1 = b0.add(&tr::literal(1));
    b1.schedule();

    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    // dbg!(&b1.data().buffer().unwrap().to_host::<i32>().unwrap());
    assert_eq!(b1.to_vec::<i32>(), vec![2, 2, 2, 2, 2]);
}
#[test]
fn scatter_chain2() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(0, 5);
    let b = a.add(&tr::literal(1));
    tr::literal(1).scatter(&a, &tr::index(5));

    b.schedule();

    let graph = tr::compile();
    dbg!(&graph);
    graph.launch(&device);

    dbg!(b.to_vec::<i32>());
    dbg!(a.to_vec::<i32>());
    assert_eq!(b.to_vec::<i32>(), vec![2, 2, 2, 2, 2]);
    assert_eq!(a.to_vec::<i32>(), vec![1, 1, 1, 1, 1]);
}
#[test]
fn extract() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(1f32, 10);
    let b = tr::sized_literal(2f32, 10);

    let v = tr::vec(&[&a, &b]);

    let a = v.extract(0);
    let b = v.extract(1);

    v.schedule();
    a.schedule();
    b.schedule();

    tr::compile().launch(&device);

    dbg!(v.to_vec::<f32>());
    dbg!(a.to_vec::<f32>());
    dbg!(b.to_vec::<f32>());
}
#[test]
fn extract2() {
    pretty_env_logger::try_init().ok();
    let device = vulkan(0);

    let b = tr::sized_literal(0xffu8, 2);
    let a = tr::sized_literal(2u32, 2);

    let s = tr::composite(&[&a, &b]);

    s.schedule();

    tr::compile().launch(&device);

    dbg!(&s.to_vec::<u8>());
}
#[test]
fn test_struct() {
    let device = backend::Device::vulkan(0).unwrap();

    let a = tr::sized_literal(1u8, 10);
    let b = tr::sized_literal(2u32, 10);

    let s = tr::composite(&[&a, &b]);

    let a = s.extract(0);
    let b = s.extract(1);

    s.schedule();
    a.schedule();
    b.schedule();

    tr::compile().launch(&device);

    dbg!(s.to_vec::<u8>());
    dbg!(a.to_vec::<u8>());
    dbg!(b.to_vec::<u32>());
}

#[test]
fn texture() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let b = tr::sized_literal(1f32, 10 * 10 * 4);

    let tex = b.texture(&[10, 10], 4);

    let x = tr::sized_literal(0.5f32, 2);
    let y = tr::sized_literal(0.5f32, 2);
    // let z = tr::sized_literal(0.5f32, 2);

    let v = tex.tex_lookup(&[&x, &y]);

    v.schedule();

    tr::compile().launch(&device);

    tr::with_trace(|trace| {
        dbg!(&trace);
    });
    dbg!(v.to_vec::<f32>());
}
#[test]
fn conditionals() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let dst = tr::sized_literal(true, 100);

    dst.schedule();

    let graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    dbg!(&dst.to_vec::<u8>());
}
#[test]
fn conditional_scatter() {
    pretty_env_logger::try_init().ok();
    let device = backend::Device::vulkan(0).unwrap();

    let dst = tr::sized_literal(0, 10);
    let active = tr::array(
        &[
            true, true, false, false, true, false, true, false, true, false,
        ],
        &device,
    );
    dbg!(&active.to_vec::<u8>());

    tr::literal(1).scatter_if(&dst, &tr::index(10), &active);

    dst.schedule();

    let graph = tr::compile();
    insta::assert_debug_snapshot!(graph);
    graph.launch(&device);

    dbg!(&dst.to_vec::<i32>());
}
