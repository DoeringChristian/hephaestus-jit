use crate::var::Var;
use jit;
use std::marker::PhantomData;

pub struct Texture<const D: usize, T> {
    tex: Var<T>,
    shape: [usize; D],
    channels: usize,
}

impl<const D: usize, T: jit::AsVarType> Texture<D, T> {
    pub fn from_data(data: impl Into<Var<T>>, shape: [usize; D], channels: usize) -> Self {
        let tex = Var(data.into().texture(&shape, channels), PhantomData);

        Self {
            tex,
            shape,
            channels,
        }
    }
}
impl<T: jit::AsVarType> Texture<1, T> {
    pub fn lookup(&self, pos: impl AsRef<Var<f32>>) -> impl Iterator<Item = Var<T>> {
        self.tex
            .0
            .tex_lookup(&pos.as_ref().0)
            .extract_all()
            .map(|r| Var::<T>(r, PhantomData))
    }
}

impl<T: jit::AsVarType> Texture<2, T> {
    pub fn lookup(&self, pos: impl AsRef<crate::Vector2<f32>>) -> impl Iterator<Item = Var<T>> {
        self.tex
            .0
            .tex_lookup(&pos.as_ref().ravel())
            .extract_all()
            .map(|r| Var::<T>(r, PhantomData))
    }
}

impl<T: jit::AsVarType> Texture<3, T> {
    pub fn lookup(&self, pos: impl Into<crate::Vector3<f32>>) -> impl Iterator<Item = Var<T>> {
        self.tex
            .0
            .tex_lookup(&pos.into().ravel())
            .extract_all()
            .map(|r| Var::<T>(r, PhantomData))
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    #[test]
    fn texture3df32() {
        let device = vulkan(0);

        let b = sized_literal(1f32, 10 * 10 * 10 * 4);
        let tex = Texture::from_data(b, [10, 10, 10], 4);

        let x = sized_literal(0.5f32, 2);
        let y = literal(0.3f32);
        let z = literal(0.5f32);
        let pos = vec3(x, y, z);

        let res = tex.lookup(pos).collect::<Vec<_>>();

        res[0].schedule();

        let graph = compile().unwrap();
        graph.launch(&device).unwrap();

        assert_eq!(res[0].to_vec(), vec![1.0; 2]);
    }
}
