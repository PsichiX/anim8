use crate::{
    Scalar,
    curve::{Curve, CurveError, Curved, CurvedChange},
    utils::range_iter,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{convert::TryFrom, error::Error};

const EPSILON: Scalar = Scalar::EPSILON * 10.0;

/// Defines spline point direction.
///
/// You can think of it as tangent vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplinePointDirection<T>
where
    T: Curved,
{
    /// Tangent vector mirrors in both directions going out from the point.
    Single(T),
    /// Separate in and out tangent vectors for given point.
    InOut(T, T),
}

impl<T> Default for SplinePointDirection<T>
where
    T: Curved,
{
    fn default() -> Self {
        Self::Single(T::zero())
    }
}

impl<T> SplinePointDirection<T>
where
    T: Curved,
{
    pub fn reverse(&self) -> Self {
        match self {
            Self::Single(value) => Self::Single(value.negate()),
            Self::InOut(value_in, value_out) => Self::InOut(value_out.negate(), value_in.negate()),
        }
    }
}

impl<T> PartialEq for SplinePointDirection<T>
where
    T: Curved + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Single(l0), Self::Single(r0)) => l0 == r0,
            (Self::InOut(l0, l1), Self::InOut(r0, r1)) => l0 == r0 && l1 == r1,
            _ => false,
        }
    }
}

/// Defines spline point.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SplinePoint<T>
where
    T: Curved,
{
    pub point: T,
    #[serde(default)]
    pub direction: SplinePointDirection<T>,
}

impl<T> SplinePoint<T>
where
    T: Curved,
{
    pub fn point(point: T) -> Self {
        Self {
            point,
            direction: Default::default(),
        }
    }

    pub fn new(point: T, direction: SplinePointDirection<T>) -> Self {
        Self { point, direction }
    }

    /// Produces list of spline points going through series of points.
    ///
    /// Uses Catmull-Rom method of finding valid tangents for spline points smooth continuity.
    pub fn smooth(points: &[T], strength: Scalar) -> Vec<Self>
    where
        T: Clone + CurvedChange,
    {
        (0..points.len())
            .map(|index| {
                let prev = if index == 0 {
                    None
                } else {
                    Some(points[index - 1].clone())
                };
                let curr = points[index].clone();
                let next = if index < points.len() - 1 {
                    Some(points[index + 1].clone())
                } else {
                    None
                };
                match (prev, curr, next) {
                    (None, a, None) => SplinePoint::point(a),
                    (Some(a), b, None) => {
                        let dir = a.delta(&b).scale(0.5 * strength);
                        SplinePoint::new(b, SplinePointDirection::InOut(T::zero(), dir))
                    }
                    (None, a, Some(b)) => {
                        let dir = a.delta(&b).scale(0.5 * strength);
                        SplinePoint::new(a, SplinePointDirection::InOut(dir, T::zero()))
                    }
                    (Some(a), b, Some(c)) => {
                        let dir_ab = a.delta(&b);
                        let dir_bc = b.delta(&c);
                        let len_ab = dir_ab.length();
                        let len_bc = dir_bc.length();
                        let tangent = dir_ab
                            .normalize()
                            .offset(&dir_bc.normalize())
                            .negate()
                            .scale(0.25);
                        SplinePoint::new(
                            b,
                            SplinePointDirection::InOut(
                                tangent.scale(len_ab * strength),
                                tangent.scale(len_bc * strength),
                            ),
                        )
                    }
                }
            })
            .collect()
    }

    pub fn from_curve(curve: &Curve<T>) -> [Self; 2]
    where
        T: Clone + Curved + CurvedChange,
    {
        [
            Self::new(
                curve.from().clone(),
                SplinePointDirection::Single(curve.from().delta(curve.from_param())),
            ),
            Self::new(
                curve.to().clone(),
                SplinePointDirection::Single(curve.to_param().delta(curve.to())),
            ),
        ]
    }

    pub fn reverse(&self) -> Self
    where
        T: Clone,
    {
        Self::new(self.point.clone(), self.direction.reverse())
    }

    pub fn is_similar_to(&self, other: &Self) -> bool
    where
        T: Clone + Curved + CurvedChange,
    {
        if self.point.delta(&other.point).length_squared() >= EPSILON {
            return false;
        }
        let (a, b) = match &self.direction {
            SplinePointDirection::Single(dir) => (dir.negate(), dir.clone()),
            SplinePointDirection::InOut(prev, next) => (prev.clone(), next.negate()),
        };
        let (c, d) = match &other.direction {
            SplinePointDirection::Single(dir) => (dir.negate(), dir.clone()),
            SplinePointDirection::InOut(prev, next) => (prev.clone(), next.negate()),
        };
        a.delta(&c).length_squared() < EPSILON && b.delta(&d).length_squared() < EPSILON
    }
}

impl<T> From<T> for SplinePoint<T>
where
    T: Curved,
{
    fn from(value: T) -> Self {
        Self::point(value)
    }
}

impl<T> From<(T, T)> for SplinePoint<T>
where
    T: Curved,
{
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, SplinePointDirection::Single(value.1))
    }
}

impl<T> From<(T, T, T)> for SplinePoint<T>
where
    T: Curved,
{
    fn from(value: (T, T, T)) -> Self {
        Self::new(value.0, SplinePointDirection::InOut(value.1, value.2))
    }
}

impl<T> From<[T; 2]> for SplinePoint<T>
where
    T: Curved,
{
    fn from(value: [T; 2]) -> Self {
        let [a, b] = value;
        Self::new(a, SplinePointDirection::Single(b))
    }
}

impl<T> From<[T; 3]> for SplinePoint<T>
where
    T: Curved,
{
    fn from(value: [T; 3]) -> Self {
        let [a, b, c] = value;
        Self::new(a, SplinePointDirection::InOut(b, c))
    }
}

impl<T> PartialEq for SplinePoint<T>
where
    T: Curved + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.direction == other.direction
    }
}

/// Errors happening within spline operations.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SplineError {
    EmptyPointsList,
    Curve(
        /// Curve index (matches points pair index).
        usize,
        /// Curve error.
        CurveError,
    ),
    CurvePair(
        /// First curve index (matches points pair index).
        usize,
        /// Second curve index (matches points pair index).
        usize,
        /// Curve error.
        CurveError,
    ),
}

impl std::fmt::Display for SplineError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::EmptyPointsList => write!(f, "Empty points list"),
            Self::Curve(index, error) => write!(f, "Curve #{} error: {}", index, error),
            Self::CurvePair(first_index, second_index, error) => write!(
                f,
                "Curve #{} with curve #{} error: {}",
                first_index, second_index, error
            ),
        }
    }
}

impl Error for SplineError {}

/// Serializable spline definition defined by spline points.
pub type SplineDef<T> = Vec<SplinePoint<T>>;

/// Interpolated Bezier Spline.
///
/// Builds set of Interpolated Bezier Curves out of spline points.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "SplineDef<T>")]
#[serde(into = "SplineDef<T>")]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub struct Spline<T>
where
    T: Default + Clone + Curved + CurvedChange,
{
    points: Vec<SplinePoint<T>>,
    cached: Vec<Curve<T>>,
    length: Scalar,
    points_distances_values: Vec<(Scalar, T)>,
}

impl<T> Default for Spline<T>
where
    T: Default + Clone + Curved + CurvedChange,
{
    fn default() -> Self {
        Self::point(T::zero()).unwrap()
    }
}

impl<T> Spline<T>
where
    T: Default + Clone + Curved + CurvedChange,
{
    /// Builds spline out of spline points.
    pub fn new(mut points: Vec<SplinePoint<T>>) -> Result<Self, SplineError> {
        if points.is_empty() {
            return Err(SplineError::EmptyPointsList);
        }
        if points.len() == 1 {
            points.push(points[0].clone())
        }
        let cached = points
            .windows(2)
            .enumerate()
            .map(|(index, pair)| {
                let from_direction = match &pair[0].direction {
                    SplinePointDirection::Single(dir) => dir.clone(),
                    SplinePointDirection::InOut(_, dir) => dir.negate(),
                };
                let to_direction = match &pair[1].direction {
                    SplinePointDirection::Single(dir) => dir.negate(),
                    SplinePointDirection::InOut(dir, _) => dir.clone(),
                };
                let from_param = pair[0].point.offset(&from_direction);
                let to_param = pair[1].point.offset(&to_direction);
                Curve::bezier(
                    pair[0].point.clone(),
                    from_param,
                    to_param,
                    pair[1].point.clone(),
                )
                .map_err(|error| SplineError::Curve(index, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lengths = cached
            .iter()
            .map(|curve| curve.length())
            .collect::<Vec<_>>();
        let mut distance = 0.0;
        let mut points_distances_values = Vec::with_capacity(points.len());
        points_distances_values.push((0.0, points[0].point.clone()));
        for (length, point) in lengths.iter().zip(points.iter().skip(1)) {
            distance += length;
            points_distances_values.push((distance, point.point.clone()));
        }
        Ok(Self {
            points,
            cached,
            length: distance,
            points_distances_values,
        })
    }

    /// Builds linear spline out fo two end points.
    pub fn linear(from: T, to: T) -> Result<Self, SplineError> {
        Self::new(vec![SplinePoint::point(from), SplinePoint::point(to)])
    }

    /// Builds zero length spline out of single point.
    pub fn point(point: T) -> Result<Self, SplineError> {
        Self::new(vec![SplinePoint::point(point)])
    }

    /// Builds smooth spline going through series of points.
    ///
    /// Uses Catmull-Rom method of finding valid tangents for spline smooth continuity.
    pub fn smooth(points: &[T], strength: Scalar) -> Result<Self, SplineError> {
        Self::new(SplinePoint::smooth(points, strength))
    }

    /// Reverse spline so all points build a sequence from last to first with reversed directions.
    pub fn reverse(&self) -> Result<Self, SplineError> {
        Self::new(
            self.points
                .iter()
                .rev()
                .map(|point| point.reverse())
                .collect(),
        )
    }

    /// Offsets this spline by distance.
    /// This produces high precision offsetted spline that ensures spline follow
    /// original shape as closely as possible.
    pub fn offset(&self, distance: Scalar, guide: Option<&T>) -> Result<Self, SplineError> {
        let mut points = Vec::default();
        for (index, curve) in self.curves().iter().enumerate() {
            let curves = curve
                .offset(distance, guide)
                .map_err(|error| SplineError::Curve(index, error))?;
            for curve in curves {
                for point in SplinePoint::from_curve(&curve) {
                    if points
                        .last()
                        .map(|last| !point.is_similar_to(last))
                        .unwrap_or(true)
                    {
                        points.push(point);
                    }
                }
            }
        }
        Self::new(points)
    }

    /// Samples values along given axis in given number of steps.
    pub fn value_along_axis_iter(
        &self,
        steps: usize,
        axis_index: usize,
    ) -> Option<impl Iterator<Item = Scalar>> {
        let from = self.points.first()?.point.get_axis(axis_index)?;
        let to = self.points.last()?.point.get_axis(axis_index)?;
        Some(range_iter(steps, from, to))
    }

    /// Samples spline at given factor in <0; 1> range.
    pub fn sample(&self, factor: Scalar) -> T {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].sample(factor)
    }

    /// Samples spline at value of given axis.
    pub fn sample_along_axis(&self, axis_value: Scalar, axis_index: usize) -> Option<T> {
        let index = self.find_curve_index_by_axis_value(axis_value, axis_index)?;
        self.cached[index].sample_along_axis(axis_value, axis_index)
    }

    /// Samples velocity of change along the spline.
    pub fn sample_first_derivative(&self, factor: Scalar) -> T {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].sample_first_derivative(factor)
    }

    /// Samples velocity of change along the spline axis.
    pub fn sample_first_derivative_along_axis(
        &self,
        axis_value: Scalar,
        axis_index: usize,
    ) -> Option<T> {
        let index = self.find_curve_index_by_axis_value(axis_value, axis_index)?;
        self.cached[index].sample_first_derivative_along_axis(axis_value, axis_index)
    }

    /// Samples acceleration of change along the spline.
    pub fn sample_second_derivative(&self, factor: Scalar) -> T {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].sample_second_derivative(factor)
    }

    /// Samples acceleration of change along the spline axis.
    pub fn sample_second_derivative_along_axis(
        &self,
        axis_value: Scalar,
        axis_index: usize,
    ) -> Option<T> {
        let index = self.find_curve_index_by_axis_value(axis_value, axis_index)?;
        self.cached[index].sample_second_derivative_along_axis(axis_value, axis_index)
    }

    /// Sample spline K value at given factor.
    pub fn sample_k(&self, factor: Scalar) -> Scalar {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].sample_k(factor)
    }

    /// Sample curvature radius at given factor.
    pub fn sample_curvature_radius(&self, factor: Scalar) -> Scalar {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].sample_curvature_radius(factor)
    }

    /// Sample spline tangent at given factor.
    pub fn sample_tangent(&self, factor: Scalar) -> T {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].sample_tangent(factor)
    }

    /// Sample spline tangent at given axis value.
    pub fn sample_tangent_along_axis(&self, axis_value: Scalar, axis_index: usize) -> Option<T> {
        let index = self.find_curve_index_by_axis_value(axis_value, axis_index)?;
        self.cached[index].sample_tangent_along_axis(axis_value, axis_index)
    }

    /// Gets arc length of this spline.
    pub fn length(&self) -> Scalar {
        self.length
    }

    /// Gets slice of spline points.
    pub fn points(&self) -> &[SplinePoint<T>] {
        &self.points
    }

    /// Sets new spline points.
    pub fn set_points(&mut self, points: Vec<SplinePoint<T>>) {
        if let Ok(result) = Self::new(points) {
            *self = result;
        }
    }

    /// Gets slice of spline points distances and values.
    pub fn points_distances_values(&self) -> &[(Scalar, T)] {
        &self.points_distances_values
    }

    /// Gets slice of cached curves.
    pub fn curves(&self) -> &[Curve<T>] {
        &self.cached
    }

    /// Finds curve index, distance offset and factor (time) of this spline at given factor.
    pub fn find_curve_index_offset_factor(&self, mut factor: Scalar) -> (usize, Scalar, Scalar) {
        factor = factor.clamp(0.0, 1.0);
        let key = factor * self.cached.len() as Scalar;
        let index = (key as usize).min(self.cached.len().saturating_sub(1));
        let offset = self.points_distances_values[index].0;
        let factor = (key - index as Scalar).clamp(0.0, 1.0);
        (index, offset, factor)
    }

    /// Finds curve index of this spline at given axis value.
    pub fn find_curve_index_by_axis_value(
        &self,
        mut axis_value: Scalar,
        axis_index: usize,
    ) -> Option<usize> {
        let min = self.points.first().unwrap().point.get_axis(axis_index)?;
        let max = self.points.last().unwrap().point.get_axis(axis_index)?;
        axis_value = axis_value.max(min).min(max);
        let index = match self.points_distances_values.binary_search_by(|(_, value)| {
            value
                .get_axis(axis_index)
                .unwrap()
                .partial_cmp(&axis_value)
                .unwrap()
        }) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        Some(index.min(self.cached.len().saturating_sub(1)))
    }

    /// Finds distance along the spline for given time (factor).
    pub fn find_distance_for_time(&self, factor: Scalar) -> Scalar {
        let (index, offset, factor) = self.find_curve_index_offset_factor(factor);
        self.cached[index].find_distance_for_time(factor) + offset
    }

    /// Finds time (factor) for given distance along the spline.
    pub fn find_time_for_distance(&self, mut distance: Scalar) -> Scalar {
        distance = distance.max(0.0).min(self.length);
        let index = match self
            .points_distances_values
            .binary_search_by(|(d, _)| d.partial_cmp(&distance).unwrap())
        {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let index = index.min(self.cached.len().saturating_sub(1));
        let start = self.points_distances_values[index].0;
        let factor = self.cached[index].find_time_for_distance(distance - start);
        (index as Scalar + factor) / self.cached.len() as Scalar
    }

    /// Finds time (factor) for given axis value.
    pub fn find_time_for_axis(&self, axis_value: Scalar, axis_index: usize) -> Option<Scalar> {
        let index = self.find_curve_index_by_axis_value(axis_value, axis_index)?;
        self.cached[index].find_time_for_axis(axis_value, axis_index)
    }

    /// Finds best time (factor) for given estimate (guess) using provided function to iteratively
    /// calculate new estimate (guess) until reaches number of iterations or derivative gets close
    /// to no change. Usually used for Newton-Raphson method of estimation.
    pub fn find_time_for(
        &self,
        mut guess: Scalar,
        iterations: usize,
        mut f: impl FnMut(Scalar, &Self) -> Scalar,
    ) -> Scalar {
        guess = guess.clamp(0.0, 1.0);
        for _ in 0..iterations {
            let time = f(guess, self);
            if (time - guess).abs() < EPSILON {
                return time;
            }
            guess = time;
        }
        guess
    }

    /// Finds time (factor) closest to given point.
    /// Returns tuple of: (time, distance)
    pub fn find_time_closest_to_point(&self, point: &T) -> (Scalar, Scalar) {
        self.cached
            .iter()
            .map(|curve| curve.find_time_closest_to_point(point))
            .enumerate()
            .min_by(|(_, (_, a)), (_, (_, b))| a.partial_cmp(b).unwrap())
            .map(|(index, (factor, dist))| {
                (
                    (index as Scalar + factor) / self.cached.len() as Scalar,
                    dist,
                )
            })
            .unwrap()
    }

    /// Splits spline in two at factor (time) along the spline.
    /// Returns tuple of left and right parts of original spline.
    pub fn split(&self, factor: Scalar) -> Result<(Self, Self), SplineError> {
        let (index, _, factor) = self.find_curve_index_offset_factor(factor);
        let (left, right) = if factor < EPSILON {
            if index == 0 {
                return Err(SplineError::Curve(index, CurveError::CannotSplit));
            }
            (
                self.points[..(index + 1)].to_vec(),
                self.points[index..].to_vec(),
            )
        } else if factor > 1.0 - EPSILON {
            if index == self.cached.len() - 1 {
                return Err(SplineError::Curve(index, CurveError::CannotSplit));
            }
            (
                self.points[..(index + 2)].to_vec(),
                self.points[(index + 1)..].to_vec(),
            )
        } else {
            let point = self.cached[index].sample(factor);
            let tangent = self.cached[index].sample_tangent(factor);
            let left = self.points[..=index]
                .iter()
                .filter(|item| item.point.delta(&point).length_squared() > EPSILON)
                .cloned()
                .chain(std::iter::once(SplinePoint::new(
                    point.clone(),
                    SplinePointDirection::Single(tangent.clone()),
                )))
                .collect::<Vec<_>>();
            let right = std::iter::once(SplinePoint::new(
                point.clone(),
                SplinePointDirection::Single(tangent.clone()),
            ))
            .chain(
                self.points[(index + 1)..]
                    .iter()
                    .filter(|item| item.point.delta(&point).length_squared() > EPSILON)
                    .cloned(),
            )
            .collect::<Vec<_>>();
            (left, right)
        };

        Ok((Self::new(left)?, Self::new(right)?))
    }

    /// Finds list of all time (factors) at which extremities exist.
    pub fn find_extremities(&self) -> Vec<Scalar> {
        let mut result = Vec::new();
        for axis in 0..T::AXES {
            for curve in self.curves() {
                curve.find_extremities_for_axis_inner(axis, &mut result);
            }
        }
        result
    }

    /// Finds list of all time (factors) at which extremities for given axis exist.
    pub fn find_extremities_for_axis(&self, axis_index: usize) -> Vec<Scalar> {
        let mut result = Vec::new();
        for curve in self.curves() {
            curve.find_extremities_for_axis_inner(axis_index, &mut result);
        }
        result
    }

    /// Calculate AABB of the spline using its extremities.
    /// Returned value is tuple of min and max points.
    pub fn aabb(&self) -> (T, T) {
        let start = self.sample(0.0);
        let end = self.sample(1.0);
        let mut min = start.minimum(&end);
        let mut max = start.maximum(&end);
        for axis in 0..T::AXES {
            for factor in self.find_extremities_for_axis(axis) {
                let point = self.sample(factor);
                min = min.minimum(&point);
                max = max.maximum(&point);
            }
        }
        (min, max)
    }

    /// Finds list of all time (factors) pair tuples between this and other spline,
    /// at which two splines intersect.
    pub fn find_intersections(
        &self,
        other: &Self,
        max_iterations: usize,
        min_length: Scalar,
    ) -> Result<Vec<(Scalar, Scalar)>, SplineError> {
        let mut result = Default::default();
        // TODO: optimize?
        for (first, a) in self.curves().iter().enumerate() {
            for (second, b) in other.curves().iter().enumerate() {
                a.find_intersections_inner(
                    b,
                    0.0..0.5,
                    0.5..1.0,
                    max_iterations,
                    min_length,
                    &mut result,
                )
                .map_err(|error| SplineError::CurvePair(first, second, error))?;
            }
        }
        Ok(result)
    }

    /// Finds list of all time (factors) pair tuples at which this spline
    /// intersects with itself.
    pub fn find_self_intersections(
        &self,
        max_iterations: usize,
        min_length: Scalar,
    ) -> Result<Vec<(Scalar, Scalar)>, SplineError> {
        let (a, b) = self.split(0.5)?;
        a.find_intersections(&b, max_iterations, min_length)
    }
}

impl<T> TryFrom<SplineDef<T>> for Spline<T>
where
    T: Default + Clone + Curved + CurvedChange,
{
    type Error = SplineError;

    fn try_from(value: SplineDef<T>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<T> From<Spline<T>> for SplineDef<T>
where
    T: Default + Clone + Curved + CurvedChange,
{
    fn from(v: Spline<T>) -> Self {
        v.points
    }
}

impl<T> PartialEq for Spline<T>
where
    T: Default + Clone + Curved + CurvedChange + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.points == other.points
            && self.cached == other.cached
            && self.length == other.length
            && self.points_distances_values == other.points_distances_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::factor_iter;

    #[test]
    fn test_spline_split() {
        let spline = Spline::new(vec![
            SplinePoint::new((-100.0, 0.0), SplinePointDirection::Single((0.0, 100.0))),
            SplinePoint::new((0.0, 100.0), SplinePointDirection::Single((100.0, 0.0))),
            SplinePoint::new((100.0, 0.0), SplinePointDirection::Single((0.0, -100.0))),
        ])
        .unwrap();

        assert!(spline.split(0.0).is_err());
        assert!(spline.split(1.0).is_err());

        let (left, right) = spline.split(0.5).unwrap();
        assert_eq!(left.points(), &spline.points()[..=1]);
        assert_eq!(right.points(), &spline.points()[1..]);

        let (left, right) = spline.split(0.25).unwrap();
        assert_eq!(
            left.points(),
            &[
                SplinePoint {
                    point: (-100.0, 0.0),
                    direction: SplinePointDirection::Single((0.0, 100.0)),
                },
                SplinePoint {
                    point: (-87.5, 87.5),
                    #[cfg(not(feature = "scalar64"))]
                    direction: SplinePointDirection::Single((0.70710677, 0.70710677)),
                    #[cfg(feature = "scalar64")]
                    direction: SplinePointDirection::Single((
                        0.7071067811865475,
                        0.7071067811865475
                    )),
                }
            ]
        );
        assert_eq!(
            right.points(),
            &[
                SplinePoint {
                    point: (-87.5, 87.5),
                    #[cfg(not(feature = "scalar64"))]
                    direction: SplinePointDirection::Single((0.70710677, 0.70710677)),
                    #[cfg(feature = "scalar64")]
                    direction: SplinePointDirection::Single((
                        0.7071067811865475,
                        0.7071067811865475
                    )),
                },
                SplinePoint {
                    point: (0.0, 100.0),
                    direction: SplinePointDirection::Single((100.0, 0.0)),
                },
                SplinePoint {
                    point: (100.0, 0.0),
                    direction: SplinePointDirection::Single((0.0, -100.0)),
                }
            ]
        );

        let (left, right) = spline.split(0.75).unwrap();
        assert_eq!(
            left.points(),
            &[
                SplinePoint {
                    point: (-100.0, 0.0),
                    direction: SplinePointDirection::Single((0.0, 100.0)),
                },
                SplinePoint {
                    point: (0.0, 100.0),
                    direction: SplinePointDirection::Single((100.0, 0.0)),
                },
                SplinePoint {
                    point: (87.5, 87.5),
                    #[cfg(not(feature = "scalar64"))]
                    direction: SplinePointDirection::Single((0.70710677, -0.70710677)),
                    #[cfg(feature = "scalar64")]
                    direction: SplinePointDirection::Single((
                        0.7071067811865475,
                        -0.7071067811865475
                    )),
                }
            ]
        );
        assert_eq!(
            right.points(),
            &[
                SplinePoint {
                    point: (87.5, 87.5),
                    #[cfg(not(feature = "scalar64"))]
                    direction: SplinePointDirection::Single((0.70710677, -0.70710677)),
                    #[cfg(feature = "scalar64")]
                    direction: SplinePointDirection::Single((
                        0.7071067811865475,
                        -0.7071067811865475
                    )),
                },
                SplinePoint {
                    point: (100.0, 0.0),
                    direction: SplinePointDirection::Single((0.0, -100.0)),
                }
            ]
        );
    }

    #[test]
    fn test_spline_point_similarity() {
        let a = SplinePoint {
            point: (4.0, 2.0),
            direction: SplinePointDirection::Single((1.0, 0.0)),
        };
        let b = SplinePoint {
            point: (0.0, 0.0),
            direction: SplinePointDirection::Single((1.0, 0.0)),
        };
        let c = SplinePoint {
            point: (4.0, 2.0),
            direction: SplinePointDirection::Single((0.0, 1.0)),
        };
        assert!(a.is_similar_to(&a));
        assert!(!a.is_similar_to(&b));
        assert!(!a.is_similar_to(&c));
    }

    #[test]
    fn test_spline_offset() {
        const DISTANCE: Scalar = 10.0;

        let spline = Spline::new(vec![
            SplinePoint::new((0.0, 0.0), SplinePointDirection::Single((50.0, 0.0))),
            SplinePoint::new((100.0, 100.0), SplinePointDirection::Single((0.0, 50.0))),
        ])
        .unwrap();
        let offsetted = spline.offset(DISTANCE, None).unwrap();
        for factor in factor_iter(10) {
            let a = spline.sample(factor);
            let b = offsetted.sample(factor);
            let difference = a.delta(&b).length();
            assert!(
                difference.is_nearly_equal_to(&10.0, 1.0),
                "difference: {} at factor: {}",
                difference,
                factor * 0.5
            );
        }

        let spline = Spline::new(vec![
            SplinePoint::new((0.0, 0.0), SplinePointDirection::Single((100.0, 0.0))),
            SplinePoint::new((0.0, 100.0), SplinePointDirection::Single((-100.0, 0.0))),
        ])
        .unwrap();
        let offsetted = spline.offset(DISTANCE, None).unwrap();
        for factor in factor_iter(10) {
            let a = spline.sample(factor);
            let b = offsetted.sample(factor);
            let difference = a.delta(&b).length();
            assert!(
                difference.is_nearly_equal_to(&10.0, 1.0),
                "difference: {} at factor: {}",
                difference,
                factor * 0.5
            );
        }
    }
}
