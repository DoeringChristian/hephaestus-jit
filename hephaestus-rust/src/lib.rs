mod accel;
mod matrix;
mod point;
mod texture;
mod traits;
pub mod var;
mod vector;

pub use jit;
pub use jit::{
    compile, record, recorded, vulkan, AsVarType, Construct, Device, DynHash, Graph, ReduceOp,
    Traverse,
};

pub use accel::{Accel, AccelDesc, GeometryDesc};
pub use matrix::{
    Matrix3, Matrix3d, Matrix3f, Matrix3i, Matrix3u, Matrix4, Matrix4d, Matrix4f, Matrix4i,
    Matrix4u,
};
pub use point::{
    point2, point3, Point2, Point2d, Point2f, Point2i, Point2u, Point3, Point3d, Point3f, Point3i,
    Point3u,
};
pub use texture::Texture;
pub use traits::{Gather, Scatter};
pub use var::{arr, array, composite, dyn_index, index, literal, sized_index, sized_literal, Var};
pub use var::{
    Float16, Float32, Float64, Int16, Int32, Int64, Int8, Mask, UInt16, UInt32, UInt64, UInt8,
};
pub use vector::{
    vec2, vec3, vec4, Vector2, Vector2d, Vector2f, Vector2i, Vector2u, Vector3, Vector3d, Vector3f,
    Vector3i, Vector3u, Vector4, Vector4d, Vector4f, Vector4i, Vector4u,
};
