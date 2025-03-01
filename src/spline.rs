use crate::{
    curve::{Curve, CurveError, Curved, CurvedChange},
    utils::range_iter,
    Scalar,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
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

    pub fn reverse(&self) -> Self
    where
        T: Clone,
    {
        Self::new(self.point.clone(), self.direction.reverse())
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

/// Errors happening within spline operations.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SplineError {
    EmptyPointsList,
    Curve(
        /// Points pair index.
        usize,
        /// Curve error.
        CurveError,
    ),
}

impl std::fmt::Display for SplineError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
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
        axis_value = axis_value.clamp(min, max);
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
        distance = distance.clamp(0.0, self.length);
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
