use crate::{utils::range_iter, Scalar};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, RwLock},
};

const EPSILON: Scalar = Scalar::EPSILON * 10.0;
const NEWTON_RAPHSON_ITERATIONS: usize = 7;
const ARC_LENGTH_ITERATIONS: usize = 5;

/// Curved trait gives an interface over Interpolated Bezier Curve.
pub trait Curved {
    fn zero() -> Self;
    fn one() -> Self;
    fn negate(&self) -> Self;
    fn scale(&self, value: Scalar) -> Self;
    fn inverse_scale(&self, value: Scalar) -> Self;
    fn length(&self) -> Scalar;
    fn length_squared(&self) -> Scalar;
    fn get_axis(&self, index: usize) -> Option<Scalar>;
    fn set_axis(&mut self, index: usize, value: Scalar);
    fn count_axes(&self) -> usize;
    fn interpolate(&self, other: &Self, factor: Scalar) -> Self;
    fn is_valid(&self) -> bool;

    fn normalize(&self) -> Self
    where
        Self: Sized,
    {
        let length = self.length();
        if length > 0.0 {
            self.inverse_scale(length)
        } else {
            Self::zero()
        }
    }

    fn perpendicular(&self, _guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }
}

impl Curved for Scalar {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn negate(&self) -> Self {
        -self
    }

    fn scale(&self, value: Scalar) -> Self {
        self * value
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        self / value
    }

    fn length(&self) -> Scalar {
        self.abs()
    }

    fn length_squared(&self) -> Scalar {
        self * self
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        match index {
            0 => Some(*self),
            _ => None,
        }
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        if index == 0 {
            *self = value;
        }
    }

    fn count_axes(&self) -> usize {
        1
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let diff = other - self;
        diff * factor + self
    }

    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl Curved for (Scalar, Scalar) {
    fn zero() -> Self {
        (0.0, 0.0)
    }

    fn one() -> Self {
        (1.0, 1.0)
    }

    fn negate(&self) -> Self {
        (-self.0, -self.1)
    }

    fn scale(&self, value: Scalar) -> Self {
        (self.0 * value, self.1 * value)
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        (self.0 / value, self.1 / value)
    }

    fn length(&self) -> Scalar {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> Scalar {
        self.0 * self.0 + self.1 * self.1
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        match index {
            0 => Some(self.0),
            1 => Some(self.1),
            _ => None,
        }
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        match index {
            0 => {
                self.0 = value;
            }
            1 => {
                self.1 = value;
            }
            _ => {}
        }
    }

    fn count_axes(&self) -> usize {
        2
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let diff0 = other.0 - self.0;
        let diff1 = other.1 - self.1;
        (diff0 * factor + self.0, diff1 * factor + self.1)
    }

    fn is_valid(&self) -> bool {
        self.0.is_valid() && self.1.is_valid()
    }

    fn perpendicular(&self, _guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        Some((self.1, -self.0))
    }
}

impl Curved for [Scalar; 2] {
    fn zero() -> Self {
        [0.0, 0.0]
    }

    fn one() -> Self {
        [1.0, 1.0]
    }

    fn negate(&self) -> Self {
        [-self[0], -self[1]]
    }

    fn scale(&self, value: Scalar) -> Self {
        [self[0] * value, self[1] * value]
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        [self[0] / value, self[1] / value]
    }

    fn length(&self) -> Scalar {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> Scalar {
        self[0] * self[0] + self[1] * self[1]
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        match index {
            0 => Some(self[0]),
            1 => Some(self[1]),
            _ => None,
        }
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        match index {
            0 => {
                self[0] = value;
            }
            1 => {
                self[1] = value;
            }
            _ => {}
        }
    }

    fn count_axes(&self) -> usize {
        2
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let diff0 = other[0] - self[0];
        let diff1 = other[1] - self[1];
        [diff0 * factor + self[0], diff1 * factor + self[1]]
    }

    fn is_valid(&self) -> bool {
        self[0].is_valid() && self[1].is_valid()
    }

    fn perpendicular(&self, _guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        Some([self[1], -self[0]])
    }
}

impl Curved for (Scalar, Scalar, Scalar) {
    fn zero() -> Self {
        (0.0, 0.0, 0.0)
    }

    fn one() -> Self {
        (1.0, 1.0, 1.0)
    }

    fn negate(&self) -> Self {
        (-self.0, -self.1, -self.2)
    }

    fn scale(&self, value: Scalar) -> Self {
        (self.0 * value, self.1 * value, self.2 * value)
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        (self.0 / value, self.1 / value, self.2 / value)
    }

    fn length(&self) -> Scalar {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> Scalar {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        match index {
            0 => Some(self.0),
            1 => Some(self.1),
            2 => Some(self.2),
            _ => None,
        }
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        match index {
            0 => {
                self.0 = value;
            }
            1 => {
                self.1 = value;
            }
            2 => {
                self.2 = value;
            }
            _ => {}
        }
    }

    fn count_axes(&self) -> usize {
        3
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let diff0 = other.0 - self.0;
        let diff1 = other.1 - self.1;
        let diff2 = other.2 - self.2;
        (
            diff0 * factor + self.0,
            diff1 * factor + self.1,
            diff2 * factor + self.2,
        )
    }

    fn is_valid(&self) -> bool {
        self.0.is_valid() && self.1.is_valid() && self.2.is_valid()
    }

    fn perpendicular(&self, guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        let guide = guide?;
        Some((
            (self.1 * guide.2) - (self.2 * guide.1),
            (self.2 * guide.0) - (self.0 * guide.2),
            (self.0 * guide.1) - (self.1 * guide.0),
        ))
    }
}

impl Curved for [Scalar; 3] {
    fn zero() -> Self {
        [0.0, 0.0, 0.0]
    }

    fn one() -> Self {
        [1.0, 1.0, 1.0]
    }

    fn negate(&self) -> Self {
        [-self[0], -self[1], self[2]]
    }

    fn scale(&self, value: Scalar) -> Self {
        [self[0] * value, self[1] * value, self[2] * value]
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        [self[0] / value, self[1] / value, self[2] / value]
    }

    fn length(&self) -> Scalar {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> Scalar {
        self[0] * self[0] + self[1] * self[1] + self[2] * self[2]
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        match index {
            0 => Some(self[0]),
            1 => Some(self[1]),
            2 => Some(self[2]),
            _ => None,
        }
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        match index {
            0 => {
                self[0] = value;
            }
            1 => {
                self[1] = value;
            }
            2 => {
                self[2] = value;
            }
            _ => {}
        }
    }

    fn count_axes(&self) -> usize {
        3
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let diff0 = other[0] - self[0];
        let diff1 = other[1] - self[1];
        let diff2 = other[2] - self[2];
        [
            diff0 * factor + self[0],
            diff1 * factor + self[1],
            diff2 * factor + self[2],
        ]
    }

    fn is_valid(&self) -> bool {
        self[0].is_valid() && self[1].is_valid() && self[2].is_valid()
    }

    fn perpendicular(&self, guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        let guide = guide?;
        Some([
            (self[1] * guide[2]) - (self[2] * guide[1]),
            (self[2] * guide[0]) - (self[0] * guide[2]),
            (self[0] * guide[1]) - (self[1] * guide[0]),
        ])
    }
}

impl<T> Curved for Rc<RefCell<T>>
where
    T: Curved,
{
    fn zero() -> Self {
        Rc::new(RefCell::new(T::zero()))
    }

    fn one() -> Self {
        Rc::new(RefCell::new(T::one()))
    }

    fn negate(&self) -> Self {
        Rc::new(RefCell::new(self.borrow().negate()))
    }

    fn scale(&self, value: Scalar) -> Self {
        Rc::new(RefCell::new(self.borrow().scale(value)))
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        Rc::new(RefCell::new(self.borrow().inverse_scale(value)))
    }

    fn length(&self) -> Scalar {
        self.borrow().length()
    }

    fn length_squared(&self) -> Scalar {
        self.borrow().length_squared()
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        self.borrow().get_axis(index)
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        self.borrow_mut().set_axis(index, value);
    }

    fn count_axes(&self) -> usize {
        self.borrow().count_axes()
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let from: &T = &self.borrow();
        let to: &T = &other.borrow();
        let value = from.interpolate(to, factor);
        Rc::new(RefCell::new(value))
    }

    fn is_valid(&self) -> bool {
        self.borrow().is_valid()
    }

    fn perpendicular(&self, guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        let guide = guide.as_ref().map(|guide| guide.borrow());
        self.borrow()
            .perpendicular(guide.as_deref())
            .map(|value| Rc::new(RefCell::new(value)))
    }
}

impl<T> Curved for Arc<RwLock<T>>
where
    T: Curved,
{
    fn zero() -> Self {
        Arc::new(RwLock::new(T::zero()))
    }

    fn one() -> Self {
        Arc::new(RwLock::new(T::one()))
    }

    fn negate(&self) -> Self {
        Arc::new(RwLock::new(self.read().unwrap().negate()))
    }

    fn scale(&self, value: Scalar) -> Self {
        Arc::new(RwLock::new(self.read().unwrap().scale(value)))
    }

    fn inverse_scale(&self, value: Scalar) -> Self {
        Arc::new(RwLock::new(self.read().unwrap().inverse_scale(value)))
    }

    fn length(&self) -> Scalar {
        self.read().unwrap().length()
    }

    fn length_squared(&self) -> Scalar {
        self.read().unwrap().length_squared()
    }

    fn get_axis(&self, index: usize) -> Option<Scalar> {
        self.read().unwrap().get_axis(index)
    }

    fn set_axis(&mut self, index: usize, value: Scalar) {
        self.write().unwrap().set_axis(index, value);
    }

    fn count_axes(&self) -> usize {
        self.read().unwrap().count_axes()
    }

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let from: &T = &self.read().unwrap();
        let to: &T = &other.read().unwrap();
        let value = from.interpolate(to, factor);
        Arc::new(RwLock::new(value))
    }

    fn is_valid(&self) -> bool {
        self.read().unwrap().is_valid()
    }

    fn perpendicular(&self, guide: Option<&Self>) -> Option<Self>
    where
        Self: Sized,
    {
        let guide = guide.as_ref().map(|guide| guide.read().unwrap());
        self.read()
            .unwrap()
            .perpendicular(guide.as_deref())
            .map(|value| Arc::new(RwLock::new(value)))
    }
}

/// CurvedChange trait gives an interface over change in Interpolated Bezier Curve.
pub trait CurvedChange {
    fn offset(&self, other: &Self) -> Self;
    fn delta(&self, other: &Self) -> Self;
    fn dot(&self, other: &Self) -> Scalar;
}

impl CurvedChange for Scalar {
    fn offset(&self, other: &Self) -> Self {
        self + other
    }

    fn delta(&self, other: &Self) -> Self {
        other - self
    }

    fn dot(&self, other: &Self) -> Scalar {
        self * other
    }
}

impl CurvedChange for (Scalar, Scalar) {
    fn offset(&self, other: &Self) -> Self {
        (self.0 + other.0, self.1 + other.1)
    }

    fn delta(&self, other: &Self) -> Self {
        (other.0 - self.0, other.1 - self.1)
    }

    fn dot(&self, other: &Self) -> Scalar {
        self.0 * other.0 + self.1 * other.1
    }
}

impl CurvedChange for (Scalar, Scalar, Scalar) {
    fn offset(&self, other: &Self) -> Self {
        (self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }

    fn delta(&self, other: &Self) -> Self {
        (other.0 - self.0, other.1 - self.1, other.2 - self.2)
    }

    fn dot(&self, other: &Self) -> Scalar {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }
}

impl CurvedChange for [Scalar; 2] {
    fn offset(&self, other: &Self) -> Self {
        [self[0] + other[0], self[1] + other[1]]
    }

    fn delta(&self, other: &Self) -> Self {
        [other[0] - self[0], other[1] - self[1]]
    }

    fn dot(&self, other: &Self) -> Scalar {
        self[0] * other[0] + self[1] * other[1]
    }
}

impl CurvedChange for [Scalar; 3] {
    fn offset(&self, other: &Self) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]]
    }

    fn delta(&self, other: &Self) -> Self {
        [other[0] - self[0], other[1] - self[1], other[2] - self[2]]
    }

    fn dot(&self, other: &Self) -> Scalar {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }
}

impl<T> CurvedChange for Rc<RefCell<T>>
where
    T: CurvedChange,
{
    fn offset(&self, other: &Self) -> Self {
        let from: &T = &self.borrow();
        let to: &T = &other.borrow();
        Rc::new(RefCell::new(from.offset(to)))
    }

    fn delta(&self, other: &Self) -> Self {
        let from: &T = &self.borrow();
        let to: &T = &other.borrow();
        Rc::new(RefCell::new(from.delta(to)))
    }

    fn dot(&self, other: &Self) -> Scalar {
        let from: &T = &self.borrow();
        let to: &T = &other.borrow();
        from.dot(to)
    }
}

impl<T> CurvedChange for Arc<RwLock<T>>
where
    T: CurvedChange,
{
    fn offset(&self, other: &Self) -> Self {
        let from: &T = &self.read().unwrap();
        let to: &T = &other.read().unwrap();
        Arc::new(RwLock::new(from.offset(to)))
    }

    fn delta(&self, other: &Self) -> Self {
        let from: &T = &self.read().unwrap();
        let to: &T = &other.read().unwrap();
        Arc::new(RwLock::new(from.delta(to)))
    }

    fn dot(&self, other: &Self) -> Scalar {
        let from: &T = &self.read().unwrap();
        let to: &T = &other.read().unwrap();
        from.dot(to)
    }
}

/// Serialziable definition of Interpolated Bezier Curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveDef<T>(
    /// P0 - starting point (`from`).
    pub T,
    /// P1 - starting tangent point (`from param`).
    pub T,
    /// P2 - ending tangent point (`to param`).
    pub T,
    /// P3 - ending point (`to`).
    pub T,
);

impl<T> Default for CurveDef<T>
where
    T: Curved,
{
    fn default() -> Self {
        Self(T::zero(), T::zero(), T::one(), T::one())
    }
}

impl<T> TryFrom<CurveDef<T>> for Curve<T>
where
    T: Clone + Curved + CurvedChange,
{
    type Error = CurveError;

    fn try_from(value: CurveDef<T>) -> Result<Self, Self::Error> {
        Self::bezier(value.0, value.1, value.2, value.3)
    }
}

impl<T> From<Curve<T>> for CurveDef<T>
where
    T: Clone + Curved + CurvedChange,
{
    fn from(v: Curve<T>) -> Self {
        Self(v.from, v.from_param, v.to_param, v.to)
    }
}

/// Curve operations errors.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum CurveError {
    InvalidFromValue,
    InvalidFromParamValue,
    InvalidToParamValue,
    InvalidToValue,
    CannotSplit,
    CannotShift,
}

impl std::fmt::Display for CurveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Interpolated Bezier Curve.
///
/// It's made out of points:
/// - P0 - starting point (`from`).
/// - P1 - starting tangent point (`from param`).
/// - P2 - ending tangent point (`to param`).
/// - P3 - ending point (`to`).
///
/// It's solved by interpolating each layer of it's points instead of using Bezier equation:
/// - A := lerp(from, from param, factor)
/// - B := lerp(from param, to param, factor)
/// - C := lerp(to param, to, factor)
/// - D := lerp(A, B, factor)
/// - E := lerp(B, C, factor)
/// - Result: lerp(D, E, factor)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "CurveDef<T>")]
#[serde(into = "CurveDef<T>")]
pub struct Curve<T>
where
    T: Clone + Curved + CurvedChange,
{
    from: T,
    from_param: T,
    to_param: T,
    to: T,
    length: Scalar,
}

impl<T> Default for Curve<T>
where
    T: Clone + Curved + CurvedChange,
{
    fn default() -> Self {
        Self::linear(T::zero(), T::one()).unwrap()
    }
}

impl<T> Curve<T>
where
    T: Clone + Curved + CurvedChange,
{
    fn new_uninitialized(from: T, from_param: T, to_param: T, to: T) -> Result<Self, CurveError> {
        if !from.is_valid() {
            return Err(CurveError::InvalidFromValue);
        }
        if !from_param.is_valid() {
            return Err(CurveError::InvalidFromParamValue);
        }
        if !to_param.is_valid() {
            return Err(CurveError::InvalidToParamValue);
        }
        if !to.is_valid() {
            return Err(CurveError::InvalidToValue);
        }
        Ok(Self {
            from,
            from_param,
            to_param,
            to,
            length: 0.0,
        })
    }

    /// Builds linear curve with `from` same as `from_param` and `to` same as `to_param`.
    pub fn linear(from: T, to: T) -> Result<Self, CurveError> {
        let from_param = from.interpolate(&to, 1.0 / 3.0);
        let to_param = from.interpolate(&to, 2.0 / 3.0);
        let mut result = Self::new_uninitialized(from, from_param, to_param, to)?;
        result.recalculate_length();
        Ok(result)
    }

    /// Builds curve from all Bezier params.
    pub fn bezier(from: T, from_param: T, to_param: T, to: T) -> Result<Self, CurveError> {
        let mut result = Self::new_uninitialized(from, from_param, to_param, to)?;
        result.recalculate_length();
        Ok(result)
    }

    fn recalculate_length(&mut self) {
        self.length = self.arc_length(EPSILON);
    }

    /// Gets starting point.
    pub fn from(&self) -> &T {
        &self.from
    }

    /// Sets starting point.
    pub fn set_from(&mut self, value: T) {
        self.from = value;
        self.recalculate_length();
    }

    /// Gets starting tangent point.
    pub fn from_param(&self) -> &T {
        &self.from_param
    }

    /// Sets starting tangent point.
    pub fn set_from_param(&mut self, value: T) {
        self.from_param = value;
        self.recalculate_length();
    }

    /// Gets ending tangent point.
    pub fn to_param(&self) -> &T {
        &self.to_param
    }

    /// Sets ending tangent point.
    pub fn set_to_param(&mut self, value: T) {
        self.to_param = value;
        self.recalculate_length();
    }

    /// Gets ending point.
    pub fn to(&self) -> &T {
        &self.to
    }

    /// Sets ending point.
    pub fn set_to(&mut self, value: T) {
        self.to = value;
        self.recalculate_length();
    }

    /// Sets all Bezier curve parameters.
    pub fn set(&mut self, from: T, from_param: T, to_param: T, to: T) {
        self.from = from;
        self.from_param = from_param;
        self.to_param = to_param;
        self.to = to;
        self.recalculate_length();
    }

    /// Gets arc length of this curve.
    pub fn length(&self) -> Scalar {
        self.length
    }

    /// Reverses curve, so: F, FP, TP, T -> T, TP, FP, F
    pub fn reverse(&self) -> Result<Self, CurveError> {
        Self::bezier(
            self.to.clone(),
            self.to_param.clone(),
            self.from_param.clone(),
            self.from.clone(),
        )
    }

    /// Shifts this curve by distance perpendicular to guide.
    /// Produces low precision shift, so use it carefully!
    pub fn shift(&self, distance: Scalar, guide: Option<&T>) -> Result<Self, CurveError> {
        let tangent_from = self.from.delta(&self.from_param).normalize();
        let tangent_to = self.to.delta(&self.to_param).normalize();
        let normal_from = tangent_from
            .perpendicular(guide)
            .ok_or(CurveError::CannotShift)?
            .normalize();
        let normal_to = tangent_to
            .perpendicular(guide)
            .ok_or(CurveError::CannotShift)?
            .normalize()
            .negate();
        let from = self.from.offset(&normal_from.scale(distance));
        let to = self.to.offset(&normal_to.scale(distance));
        let from_param = self
            .from_param
            .offset(&tangent_from.scale(distance))
            .offset(&normal_from.scale(distance));
        let to_param = self
            .to_param
            .offset(&tangent_to.scale(distance))
            .offset(&normal_to.scale(distance));
        Self::bezier(from, from_param, to_param, to)
    }

    /// Samples values along given axis in given number of steps.
    pub fn value_along_axis_iter(
        &self,
        steps: usize,
        axis_index: usize,
    ) -> Option<impl Iterator<Item = Scalar>> {
        let from = self.from.get_axis(axis_index)?;
        let to = self.to.get_axis(axis_index)?;
        Some(range_iter(steps, from, to))
    }

    /// Samples curve at given factor in <0; 1> range.
    #[allow(clippy::many_single_char_names)]
    pub fn sample(&self, mut factor: Scalar) -> T {
        factor = factor.clamp(0.0, 1.0);
        let a = self.from.interpolate(&self.from_param, factor);
        let b = self.from_param.interpolate(&self.to_param, factor);
        let c = self.to_param.interpolate(&self.to, factor);
        let d = a.interpolate(&b, factor);
        let e = b.interpolate(&c, factor);
        d.interpolate(&e, factor)
    }

    /// Samples curve at value of given axis.
    pub fn sample_along_axis(&self, axis_value: Scalar, axis_index: usize) -> Option<T> {
        let factor = self.find_time_for_axis(axis_value, axis_index)?;
        Some(self.sample(factor))
    }

    /// Samples velocity of change along the curve.
    pub fn sample_first_derivative(&self, mut factor: Scalar) -> T {
        factor = factor.clamp(0.0, 1.0);
        let a = self.from.delta(&self.from_param);
        let b = self.from_param.delta(&self.to_param);
        let c = self.to_param.delta(&self.to);
        let d = a.interpolate(&b, factor);
        let e = b.interpolate(&c, factor);
        d.interpolate(&e, factor).scale(3.0)
    }

    /// Samples velocity of change along the curve axis.
    pub fn sample_first_derivative_along_axis(
        &self,
        axis_value: Scalar,
        axis_index: usize,
    ) -> Option<T> {
        let factor = self.find_time_for_axis(axis_value, axis_index)?;
        Some(self.sample_first_derivative(factor))
    }

    /// Samples acceleration of change along the curve.
    pub fn sample_second_derivative(&self, mut factor: Scalar) -> T {
        factor = factor.clamp(0.0, 1.0);
        let a = self.from.delta(&self.from_param);
        let b = self.from_param.delta(&self.to_param);
        let c = self.to_param.delta(&self.to);
        let d = a.delta(&b);
        let e = b.delta(&c);
        d.interpolate(&e, factor).scale(6.0)
    }

    /// Samples acceleration of change along the curve axis.
    pub fn sample_second_derivative_along_axis(
        &self,
        axis_value: Scalar,
        axis_index: usize,
    ) -> Option<T> {
        let factor = self.find_time_for_axis(axis_value, axis_index)?;
        Some(self.sample_second_derivative(factor))
    }

    /// Sample curve K value at given factor.
    pub fn sample_k(&self, mut factor: Scalar) -> Scalar {
        factor = factor.clamp(0.0, 1.0);
        let first = self.sample_first_derivative(factor);
        let second = self.sample_second_derivative(factor);
        second.length() / first.length().powf(1.5)
    }

    /// Sample curvature radius at given factor.
    pub fn sample_curvature_radius(&self, factor: Scalar) -> Scalar {
        1.0 / self.sample_k(factor)
    }

    /// Sample curve tangent at given factor.
    pub fn sample_tangent(&self, mut factor: Scalar) -> T {
        factor = factor.clamp(EPSILON, 1.0 - EPSILON);
        let direction = self.sample_first_derivative(factor);
        let length = direction.length();
        direction.inverse_scale(length)
    }

    /// Sample curve tangent at given axis value.
    pub fn sample_tangent_along_axis(&self, axis_value: Scalar, axis_index: usize) -> Option<T> {
        let factor = self.find_time_for_axis(axis_value, axis_index)?;
        Some(self.sample_tangent(factor))
    }

    fn split_uninitialized(&self, mut factor: Scalar) -> Result<(Self, Self), CurveError> {
        factor = factor.clamp(0.0, 1.0);
        #[allow(clippy::manual_range_contains)]
        if factor < EPSILON || factor > 1.0 - EPSILON {
            return Err(CurveError::CannotSplit);
        }
        let a = self.from.interpolate(&self.from_param, factor);
        let b = self.from_param.interpolate(&self.to_param, factor);
        let c = self.to_param.interpolate(&self.to, factor);
        let d = a.interpolate(&b, factor);
        let e = b.interpolate(&c, factor);
        let f = d.interpolate(&e, factor);
        let first = Self::new_uninitialized(self.from.clone(), a, d, f.clone())?;
        let second = Self::new_uninitialized(f, e, c, self.to.clone())?;
        Ok((first, second))
    }

    /// Splits curve into two parts at given factor.
    pub fn split(&self, factor: Scalar) -> Result<(Self, Self), CurveError> {
        self.split_uninitialized(factor).map(|(mut a, mut b)| {
            a.recalculate_length();
            b.recalculate_length();
            (a, b)
        })
    }

    fn estimate_arc_length(&self) -> Scalar {
        let a = self.from.delta(&self.from_param).length();
        let b = self.from_param.delta(&self.to_param).length();
        let c = self.to_param.delta(&self.to).length();
        let d = self.to.delta(&self.from).length();
        (a + b + c + d) * 0.5
    }

    fn arc_length(&self, threshold: Scalar) -> Scalar {
        self.arc_length_inner(self.estimate_arc_length(), threshold, ARC_LENGTH_ITERATIONS)
    }

    fn arc_length_inner(&self, estimation: Scalar, threshold: Scalar, levels: usize) -> Scalar {
        let (a, b) = match self.split_uninitialized(0.5) {
            Ok((a, b)) => (a, b),
            Err(_) => return estimation,
        };
        let ra = a.estimate_arc_length();
        let rb = b.estimate_arc_length();
        let result = ra + rb;
        if (estimation - result).abs() < threshold || levels == 0 {
            return result;
        }
        let levels = levels - 1;
        let a = a.arc_length_inner(ra, threshold, levels);
        let b = b.arc_length_inner(rb, threshold, levels);
        a + b
    }

    /// Finds distance along the curve for given time (factor).
    pub fn find_distance_for_time(&self, factor: Scalar) -> Scalar {
        self.split_uninitialized(factor)
            .map(|(mut curve, _)| {
                curve.recalculate_length();
                curve.length
            })
            .unwrap_or_else(|_| if factor < 0.5 { 0.0 } else { self.length })
    }

    /// Finds time (factor) for given distance along the curve.
    pub fn find_time_for_distance(&self, mut distance: Scalar) -> Scalar {
        if self.length < EPSILON {
            return 0.0;
        }
        distance = distance.clamp(0.0, self.length);
        self.find_time_for(
            None,
            None,
            |time| {
                let dv = self.find_distance_for_time(time) - distance;
                let tangent = self.sample_tangent(time);
                Some(tangent.scale(dv))
            },
            |_| true,
        )
    }

    /// Finds time (factor) for given axis value.
    pub fn find_time_for_axis(&self, mut axis_value: Scalar, axis_index: usize) -> Option<Scalar> {
        let min = self.from.get_axis(axis_index)?;
        let max = self.to.get_axis(axis_index)?;
        let dist = max - min;
        if dist.abs() < EPSILON {
            return Some(1.0);
        }
        axis_value = axis_value.clamp(min, max);
        Some(self.find_time_for(
            None,
            None,
            |time| {
                let dv = self.sample(time).get_axis(axis_index)? - axis_value;
                let tangent = self.sample_tangent(time);
                Some(tangent.scale(dv))
            },
            |_| true,
        ))
    }

    /// Finds time (factor) closest to given point.
    /// Returns tuple of: (time, distance)
    pub fn find_time_closest_to_point(&self, point: &T) -> (Scalar, Scalar) {
        let mut lowest_distance = Scalar::INFINITY;
        let time = self.find_time_for(
            None,
            None,
            |time| Some(point.delta(&self.sample(time))),
            |time| {
                let distance = self.sample(time).delta(point).length();
                if distance < lowest_distance {
                    lowest_distance = distance;
                    true
                } else {
                    false
                }
            },
        );
        (time, self.sample(time).delta(point).length())
    }

    /// Finds best time (factor) for given estimate (guess) using provided function to
    /// calculate change in hidden value, until reaches number of iterations or derivative
    /// gets close to no change. Usually used for Newton-Raphson method of approximation.
    pub fn find_time_for(
        &self,
        guess: Option<Scalar>,
        iterations: Option<usize>,
        mut difference: impl FnMut(Scalar) -> Option<T>,
        mut validation: impl FnMut(Scalar) -> bool,
    ) -> Scalar {
        let mut change = |time| {
            let diff = difference(time)?;
            let d1 = self.sample_first_derivative(time);
            let d2 = self.sample_second_derivative(time);
            let d1_sqr = d1.dot(&d1);
            let c = diff.dot(&d2);
            let fitness = 2.0 * diff.dot(&d1);
            let fitness_derivative = 2.0 * (d1_sqr + c);
            Some((fitness, fitness_derivative))
        };
        let mut guess = guess.unwrap_or(0.5).clamp(0.0, 1.0);
        let iterations = iterations.unwrap_or(NEWTON_RAPHSON_ITERATIONS);
        for _ in 0..iterations {
            let Some((fitness, fitness_derivative)) = change(guess) else {
                break;
            };
            if fitness.abs() < EPSILON {
                break;
            }
            if fitness_derivative.abs() < EPSILON {
                break;
            }
            let time = (guess - fitness / fitness_derivative).clamp(0.0, 1.0);
            if (guess - time).abs() < EPSILON {
                break;
            }
            if !validation(time) {
                break;
            }
            guess = time;
        }
        guess
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::factor_iter;

    #[test]
    fn test_curve_approximation() {
        let curve = Curve::bezier((0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)).unwrap();

        for x in factor_iter(100) {
            let time = curve.find_time_for_axis(x, 0).unwrap();
            let sample = curve.sample(time);
            let diff = (sample.0 - x).abs();
            assert!(
                diff < 1.0e-3,
                "x: {} | time: {} | sample: {:?} | difference: {}",
                x,
                time,
                sample,
                diff
            );
        }

        for time in factor_iter(100) {
            let distance = curve.find_distance_for_time(time);
            let sample = curve.find_time_for_distance(distance);
            let diff = (sample - time).abs();
            assert!(
                diff < 1.0e-3,
                "time: {} | distance: {} | sample: {:?} | difference: {}",
                time,
                distance,
                sample,
                diff
            );
        }
    }

    #[test]
    fn test_curve_shift() {
        assert!(1.0.perpendicular(None).is_none());
        assert_eq!((1.0, 2.0).perpendicular(None).unwrap(), (2.0, -1.0));
        assert_eq!(
            (1.0, 1.0, 0.0)
                .perpendicular(Some(&(0.0, 0.0, 1.0)))
                .unwrap(),
            (1.0, -1.0, 0.0)
        );

        const DISTANCE: Scalar = 10.0;

        let curve = Curve::bezier((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)).unwrap();
        let shifted = curve.shift(DISTANCE, None).unwrap();
        for factor in factor_iter(10) {
            let a = curve.sample(factor);
            let b = shifted.sample(factor);
            let distance = a.delta(&b).length();
            assert!(distance <= 10.0, "distance: {}", distance);
        }
    }

    #[test]
    fn test_curve_reverse() {
        let curve = Curve::bezier((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)).unwrap();
        let reversed = curve.reverse().unwrap();

        for factor in factor_iter(10) {
            let a = curve.sample(factor);
            let b = reversed.sample(1.0 - factor);
            let distance = a.delta(&b).length();
            assert!(distance <= 1.0e-4, "distance: {}", distance);
        }
    }
}
