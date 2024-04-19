use hephaestus as hep;

#[derive(Clone, Hash, hep::Traverse, hep::Construct)]
pub struct Instance {
    pub transform: hep::Matrix4f,
    pub geometry: hep::UInt32,
}

impl From<Instance> for hep::Var<jit::Instance> {
    fn from(value: Instance) -> Self {
        let tr = value.transform.clone();
        let transform = hep::arr([
            tr.x_axis.x,
            tr.y_axis.x,
            tr.z_axis.x,
            tr.w_axis.x, //
            tr.x_axis.y,
            tr.y_axis.y,
            tr.z_axis.y,
            tr.w_axis.y, //
            tr.x_axis.z,
            tr.y_axis.z,
            tr.z_axis.z,
            tr.w_axis.z, //
        ]);
        hep::composite()
            .elem(transform)
            .elem(value.geometry)
            .construct()
    }
}
