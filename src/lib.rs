pub mod animation;
pub mod curve;
pub mod phase;
pub mod spline;
pub mod transition;
pub mod utils;

pub mod prelude {
    pub use crate::{Scalar, animation::*, curve::*, phase::*, spline::*, transition::*, utils::*};
}

/// Scalar number type.
///
/// by default it is 32-bit flaot but you can change it to 64-bit float with `scalar64` feature.
#[cfg(not(feature = "scalar64"))]
pub type Scalar = f32;
#[cfg(feature = "scalar64")]
pub type Scalar = f64;
