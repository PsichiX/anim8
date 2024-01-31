use crate::Scalar;

/// Return iterator over uniformly spread samples of `steps` in 0 to 1 range.
pub fn factor_iter(steps: usize) -> impl Iterator<Item = Scalar> {
    (0..=steps).map(move |index| index as Scalar / steps as Scalar)
}

/// Return iterator over uniformly spread samples of `steps` in `from` to `to` range.
pub fn range_iter(steps: usize, from: Scalar, to: Scalar) -> impl Iterator<Item = Scalar> {
    let diff = to - from;
    (0..=steps).map(move |index| from + diff * index as Scalar / steps as Scalar)
}
