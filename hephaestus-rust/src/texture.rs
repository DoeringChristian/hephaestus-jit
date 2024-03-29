use crate::var::Var;
use jit;
use std::marker::PhantomData;

pub struct Texture<const D: usize, T> {
    tex: Var<T>,
    shape: [usize; D],
    channels: usize,
}

impl<const D: usize, T: jit::AsVarType> Texture<D, T> {
    pub fn from_data(data: &Var<T>, shape: &[usize], channels: usize) -> Self {
        let tex = Var(data.0.texture(&shape, channels), PhantomData);

        let mut shape_array = [0; D];
        for i in 0..D {
            shape_array[i] = shape[i];
        }

        Self {
            tex,
            shape: shape_array,
            channels,
        }
    }
}

impl<T: jit::AsVarType> Texture<2, T> {
    pub fn lookup(&self, pos: &Var<mint::Vector2<f32>>) -> impl Iterator<Item = Var<T>> {
        self.tex
            .0
            .tex_lookup(&pos.0)
            .extract_all()
            .map(|r| Var::<T>(r, PhantomData))
    }
}
