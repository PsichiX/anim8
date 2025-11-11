use crate::{Scalar, utils::range_iter};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    cmp::Ordering,
    error::Error,
    ops::Range,
    rc::Rc,
    sync::{Arc, RwLock},
};

const EPSILON: Scalar = Scalar::EPSILON * 10.0;
const NEWTON_RAPHSON_ITERATIONS: usize = 7;
const ARC_LENGTH_ITERATIONS: usize = 5;

/// Curved trait gives an interface over Interpolated Bezier Curve.
pub trait Curved {
    const AXES: usize;

    fn zero() -> Self;
    fn one() -> Self;
    fn negate(&self) -> Self;
    fn scale(&self, value: Scalar) -> Self;
    fn inverse_scale(&self, value: Scalar) -> Self;
    fn length(&self) -> Scalar;
    fn length_squared(&self) -> Scalar;
    fn get_axis(&self, index: usize) -> Option<Scalar>;
    fn set_axis(&mut self, index: usize, value: Scalar);
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
    const AXES: usize = 1;

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

    fn interpolate(&self, other: &Self, factor: Scalar) -> Self {
        let diff = other - self;
        diff * factor + self
    }

    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl Curved for (Scalar, Scalar) {
    const AXES: usize = 2;

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
    const AXES: usize = 2;

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
    const AXES: usize = 3;

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
    const AXES: usize = 3;

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
    const AXES: usize = T::AXES;

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
    const AXES: usize = T::AXES;

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
    fn minimum(&self, other: &Self) -> Self;
    fn maximum(&self, other: &Self) -> Self;

    fn is_nearly_equal_to(&self, other: &Self, epsilon: Scalar) -> bool
    where
        Self: Curved + Sized,
    {
        self.delta(other).length_squared() < epsilon * epsilon
    }
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

    fn minimum(&self, other: &Self) -> Self {
        self.min(*other)
    }

    fn maximum(&self, other: &Self) -> Self {
        self.max(*other)
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

    fn minimum(&self, other: &Self) -> Self {
        (self.0.min(other.0), self.1.min(other.1))
    }

    fn maximum(&self, other: &Self) -> Self {
        (self.0.max(other.0), self.1.max(other.1))
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

    fn minimum(&self, other: &Self) -> Self {
        (
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
        )
    }

    fn maximum(&self, other: &Self) -> Self {
        (
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
        )
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

    fn minimum(&self, other: &Self) -> Self {
        [self[0].min(other[0]), self[1].min(other[1])]
    }

    fn maximum(&self, other: &Self) -> Self {
        [self[0].max(other[0]), self[1].max(other[1])]
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

    fn minimum(&self, other: &Self) -> Self {
        [
            self[0].min(other[0]),
            self[1].min(other[1]),
            self[2].min(other[2]),
        ]
    }

    fn maximum(&self, other: &Self) -> Self {
        [
            self[0].max(other[0]),
            self[1].max(other[1]),
            self[2].max(other[2]),
        ]
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

    fn minimum(&self, other: &Self) -> Self {
        let from: &T = &self.borrow();
        let to: &T = &other.borrow();
        Rc::new(RefCell::new(from.minimum(to)))
    }

    fn maximum(&self, other: &Self) -> Self {
        let from: &T = &self.borrow();
        let to: &T = &other.borrow();
        Rc::new(RefCell::new(from.maximum(to)))
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

    fn minimum(&self, other: &Self) -> Self {
        let from: &T = &self.read().unwrap();
        let to: &T = &other.read().unwrap();
        Arc::new(RwLock::new(from.minimum(to)))
    }

    fn maximum(&self, other: &Self) -> Self {
        let from: &T = &self.read().unwrap();
        let to: &T = &other.read().unwrap();
        Arc::new(RwLock::new(from.maximum(to)))
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
    CannotOffset,
}

impl std::fmt::Display for CurveError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidFromValue => write!(f, "Invalid from value"),
            Self::InvalidFromParamValue => write!(f, "Invalid from param value"),
            Self::InvalidToParamValue => write!(f, "Invalid to param value"),
            Self::InvalidToValue => write!(f, "Invalid to value"),
            Self::CannotSplit => write!(f, "Cannot split"),
            Self::CannotShift => write!(f, "Cannot shift"),
            Self::CannotOffset => write!(f, "Cannot offset"),
        }
    }
}

impl Error for CurveError {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveCategory {
    Point,
    Straight,
    CurvedParallel,
    Curved,
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
    aabb: (T, T),
    category: CurveCategory,
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
            aabb: (T::zero(), T::zero()),
            category: CurveCategory::Curved,
        })
    }

    /// Builds linear curve with `from` same as `from_param` and `to` same as `to_param`.
    pub fn linear(from: T, to: T) -> Result<Self, CurveError> {
        let from_param = from.interpolate(&to, 1.0 / 3.0);
        let to_param = from.interpolate(&to, 2.0 / 3.0);
        let mut result = Self::new_uninitialized(from, from_param, to_param, to)?;
        result.update_cache();
        Ok(result)
    }

    /// Builds curve from all Bezier params.
    pub fn bezier(from: T, from_param: T, to_param: T, to: T) -> Result<Self, CurveError> {
        let mut result = Self::new_uninitialized(from, from_param, to_param, to)?;
        result.update_cache();
        Ok(result)
    }

    fn update_cache(&mut self) {
        self.length = self.arc_length(EPSILON);
        self.aabb = self.calculate_aabb();
        self.category = self.categorize();
    }

    /// Gets starting point.
    pub fn from(&self) -> &T {
        &self.from
    }

    /// Sets starting point.
    pub fn set_from(&mut self, value: T) {
        self.from = value;
        self.update_cache();
    }

    /// Gets starting tangent point.
    pub fn from_param(&self) -> &T {
        &self.from_param
    }

    /// Sets starting tangent point.
    pub fn set_from_param(&mut self, value: T) {
        self.from_param = value;
        self.update_cache();
    }

    /// Gets ending tangent point.
    pub fn to_param(&self) -> &T {
        &self.to_param
    }

    /// Sets ending tangent point.
    pub fn set_to_param(&mut self, value: T) {
        self.to_param = value;
        self.update_cache();
    }

    /// Gets ending point.
    pub fn to(&self) -> &T {
        &self.to
    }

    /// Sets ending point.
    pub fn set_to(&mut self, value: T) {
        self.to = value;
        self.update_cache();
    }

    /// Sets all Bezier curve parameters.
    pub fn set(&mut self, from: T, from_param: T, to_param: T, to: T) {
        self.from = from;
        self.from_param = from_param;
        self.to_param = to_param;
        self.to = to;
        self.update_cache();
    }

    /// Gets arc length of this curve.
    pub fn length(&self) -> Scalar {
        self.length
    }

    /// AABB of this curve.
    /// Returns tuple of min-max pair.
    pub fn aabb(&self) -> (&T, &T) {
        (&self.aabb.0, &self.aabb.1)
    }

    /// Returns category of this curve.
    pub fn category(&self) -> CurveCategory {
        self.category
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

    /// Projects this curve onto plane defined by origin and normal.
    pub fn to_planar(&self, plane_origin: &T, plane_normal: &T) -> Result<Self, CurveError> {
        let mut points = [
            self.from.clone(),
            self.from_param.clone(),
            self.to_param.clone(),
            self.to.clone(),
        ];
        let plane_normal = plane_normal.normalize();
        for point in &mut points {
            let v = plane_origin.delta(point);
            let distance = v.dot(&plane_normal);
            *point = plane_normal.scale(distance).delta(point);
        }
        let [from, from_param, to_param, to] = points;
        Self::bezier(from, from_param, to_param, to)
    }

    /// Shifts this curve by distance perpendicular to guide.
    /// Produces low precision shift, so use it carefully!
    /// If you need high precision, use offset function.
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

    /// Offsets this curve by distance.
    /// This produces high precision offsetted series of safe curves to ensure
    /// curves follow original shape as closely as possible.
    pub fn offset(&self, distance: Scalar, guide: Option<&T>) -> Result<Vec<Self>, CurveError> {
        let mut result = Vec::new();
        self.build_safe_curves(&mut result);
        for target in &mut result {
            let source = if let Some(guide) = guide {
                let centroid = target
                    .from
                    .offset(&target.from_param)
                    .offset(&target.to_param)
                    .offset(&target.to)
                    .scale(0.25);
                target.to_planar(&centroid, guide)?
            } else {
                target.clone()
            };
            match source.category() {
                CurveCategory::Point => {
                    return Err(CurveError::CannotOffset);
                }
                CurveCategory::Straight => {
                    let direction = source.from.delta(&source.to).normalize();
                    let Some(right) = direction.perpendicular(guide) else {
                        continue;
                    };
                    for point in [
                        &mut target.from,
                        &mut target.from_param,
                        &mut target.to_param,
                        &mut target.to,
                    ] {
                        *point = point.offset(&right.scale(distance));
                    }
                    target.update_cache();
                }
                CurveCategory::CurvedParallel => {
                    // Fallback to low precision shifting.
                    // This most likely will fail the bigger the curvature, but
                    // most likely it won't ever be the case to get curved
                    // parallel curve bc those are not safe and will be splitted.
                    *target = target.shift(distance, guide)?;
                }
                CurveCategory::Curved => {
                    let front_tangent = source.from.delta(&source.from_param).normalize();
                    let back_tangent = source.to_param.delta(&source.to).normalize();
                    let front_normal = front_tangent
                        .perpendicular(guide)
                        .ok_or(CurveError::CannotOffset)?
                        .normalize();
                    let back_normal = back_tangent
                        .perpendicular(guide)
                        .ok_or(CurveError::CannotOffset)?
                        .normalize();
                    let origin = ray_intersection(
                        &source.from,
                        &front_normal,
                        &source.to,
                        &back_normal,
                        guide,
                    )
                    .unwrap();
                    let from_param_direction = origin.delta(&source.from_param).normalize();
                    let to_param_direction = origin.delta(&source.to_param).normalize();
                    let from_point = source.from.offset(&front_normal.scale(distance));
                    let to_point = source.to.offset(&back_normal.scale(distance));
                    let from_param = ray_intersection(
                        &from_point,
                        &front_tangent,
                        &origin,
                        &from_param_direction,
                        guide,
                    )
                    .unwrap();
                    let to_param = ray_intersection(
                        &to_point,
                        &back_tangent,
                        &origin,
                        &to_param_direction,
                        guide,
                    )
                    .unwrap();
                    target.from = target.from.offset(&front_normal.scale(distance));
                    target.to = target.to.offset(&back_normal.scale(distance));
                    target.from_param = target.from.offset(&from_point.delta(&from_param));
                    target.to_param = target.to.offset(&to_point.delta(&to_param));
                    target.update_cache();
                }
            }
        }
        Ok(result)
    }

    /// Builds safe curves out of this curve.
    /// Safe curve means curve has at most one extremity and extremity is within
    /// circle between curve start and end points.
    /// Original curve is split at points of extremities until all sub-curves are
    /// considered safe.
    pub fn build_safe_curves(&self, out_result: &mut Vec<Self>) {
        let mut extremities = Vec::new();
        for axis in 0..T::AXES {
            self.find_extremities_for_axis_inner(axis, &mut extremities);
        }
        extremities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if extremities.is_empty() {
            out_result.push(self.clone());
            return;
        }
        if extremities.len() == 1 {
            let midpoint = self.sample(extremities[0]);
            let center = self.from.offset(&self.to).scale(0.5);
            let radius = self.from.delta(&self.to).length() * 0.5;
            if midpoint.delta(&center).length_squared() < radius * radius {
                out_result.push(self.clone());
                return;
            }
        }
        let mut current = Result::<Self, &Self>::Err(self);
        for factor in extremities {
            let curve = match current {
                Ok(ref curve) => curve,
                Err(curve) => curve,
            };
            let (curve, rest) = match curve.split(factor) {
                Ok(result) => result,
                Err(_) => break,
            };
            curve.build_safe_curves(out_result);
            current = Ok(rest);
        }
        out_result.push(match current {
            Ok(curve) => curve,
            Err(curve) => curve.clone(),
        });
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
            a.update_cache();
            b.update_cache();
            (a, b)
        })
    }

    /// Finds list of all time (factors) at which extremities exist.
    pub fn find_extremities(&self) -> Vec<Scalar> {
        let mut result = Vec::new();
        for axis in 0..T::AXES {
            self.find_extremities_for_axis_inner(axis, &mut result);
        }
        result
    }

    /// Finds list of all time (factors) at which extremities for given axis exist.
    pub fn find_extremities_for_axis(&self, axis_index: usize) -> Vec<Scalar> {
        let mut result = Vec::new();
        self.find_extremities_for_axis_inner(axis_index, &mut result);
        result
    }

    pub(crate) fn find_extremities_for_axis_inner(
        &self,
        axis_index: usize,
        out_result: &mut Vec<Scalar>,
    ) {
        let Some(p0) = self.from.get_axis(axis_index) else {
            return Default::default();
        };
        let Some(p1) = self.from_param.get_axis(axis_index) else {
            return Default::default();
        };
        let Some(p2) = self.to_param.get_axis(axis_index) else {
            return Default::default();
        };
        let Some(p3) = self.to.get_axis(axis_index) else {
            return Default::default();
        };
        let d0 = 3.0 * (p1 - p0);
        let d1 = 3.0 * (p2 - p1);
        let d2 = 3.0 * (p3 - p2);
        let a = d0 - 2.0 * d1 + d2;
        let b = 2.0 * (d1 - d0);
        let c = d0;
        if a.abs() < EPSILON {
            if b.abs() >= EPSILON {
                let t = -c / b;
                if t > 0.0 && t < 1.0 && !out_result.iter().any(|item| (t - item).abs() <= EPSILON)
                {
                    out_result.push(t);
                }
            }
        } else {
            let discriminant_squared = b * b - 4.0 * a * c;
            if discriminant_squared >= 0.0 {
                let discriminant = discriminant_squared.sqrt();
                let t1 = (-b + discriminant) / (2.0 * a);
                let t2 = (-b - discriminant) / (2.0 * a);
                if t1 > 0.0
                    && t1 < 1.0
                    && !out_result.iter().any(|item| (t1 - item).abs() <= EPSILON)
                {
                    out_result.push(t1);
                }
                if t2 > 0.0
                    && t2 < 1.0
                    && !out_result.iter().any(|item| (t2 - item).abs() <= EPSILON)
                {
                    out_result.push(t2);
                }
            }
        }
    }

    /// Finds list of all time (factors) pair tuples between this and other curve,
    /// at which two curves intersect.
    pub fn find_intersections(
        &self,
        other: &Self,
        max_iterations: usize,
        min_length: Scalar,
    ) -> Result<Vec<(Scalar, Scalar)>, CurveError> {
        let mut result = Default::default();
        self.find_intersections_inner(
            other,
            0.0..0.5,
            0.5..1.0,
            max_iterations,
            min_length,
            &mut result,
        )?;
        Ok(result)
    }

    pub(crate) fn find_intersections_inner(
        &self,
        other: &Self,
        range: Range<Scalar>,
        other_range: Range<Scalar>,
        mut max_iterations: usize,
        min_length: Scalar,
        out_result: &mut Vec<(Scalar, Scalar)>,
    ) -> Result<(), CurveError> {
        fn does_aabb_overlap<T: Curved + CurvedChange>(
            (a_min, a_max): (&T, &T),
            (b_min, b_max): (&T, &T),
        ) -> bool {
            for axis in 0..T::AXES {
                let Some(a_min) = a_min.get_axis(axis) else {
                    return false;
                };
                let Some(a_max) = a_max.get_axis(axis) else {
                    return false;
                };
                let Some(b_min) = b_min.get_axis(axis) else {
                    return false;
                };
                let Some(b_max) = b_max.get_axis(axis) else {
                    return false;
                };
                if a_min > b_max || a_max < b_min {
                    return false;
                }
            }
            true
        }

        if self.length() >= min_length
            && other.length() >= min_length
            && does_aabb_overlap(self.aabb(), other.aabb())
        {
            let af = (range.start + range.end) * 0.5;
            let bf = (other_range.start + other_range.end) * 0.5;
            if max_iterations > 0 {
                max_iterations -= 1;
                let (aa, ab) = self.split(0.5)?;
                let (ba, bb) = other.split(0.5)?;
                let aar = range.start..af;
                let abr = af..range.end;
                let bar = other_range.start..bf;
                let bbr = bf..other_range.end;
                aa.find_intersections_inner(
                    &ba,
                    aar.clone(),
                    bar.clone(),
                    max_iterations,
                    min_length,
                    out_result,
                )?;
                aa.find_intersections_inner(
                    &bb,
                    aar,
                    bbr.clone(),
                    max_iterations,
                    min_length,
                    out_result,
                )?;
                ab.find_intersections_inner(
                    &ba,
                    abr.clone(),
                    bar,
                    max_iterations,
                    min_length,
                    out_result,
                )?;
                ab.find_intersections_inner(&bb, abr, bbr, max_iterations, min_length, out_result)?;
            } else {
                // TODO: find line-line intersection and guiess T from that?
                // for now simple middle point should suffice.
                out_result.push((af, bf));
            }
        }
        Ok(())
    }

    /// Finds list of all time (factors) pair tuples at which this curve
    /// intersects with itself.
    pub fn find_self_intersections(
        &self,
        max_iterations: usize,
        min_length: Scalar,
    ) -> Result<Vec<(Scalar, Scalar)>, CurveError> {
        let (a, b) = self.split(0.5)?;
        a.find_intersections(&b, max_iterations, min_length)
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

    fn calculate_aabb(&self) -> (T, T) {
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

    fn categorize(&self) -> CurveCategory {
        if [&self.from_param, &self.to_param, &self.to]
            .iter()
            .all(|p| self.from.delta(p).length_squared() < EPSILON)
        {
            return CurveCategory::Point;
        }
        let direction = self.from.delta(&self.to).normalize();
        let front = self.from.delta(&self.from_param).normalize();
        let back = self.to_param.delta(&self.to).normalize();
        if front.dot(&direction).abs() >= 1.0 - EPSILON
            && back.dot(&direction).abs() >= 1.0 - EPSILON
        {
            return CurveCategory::Straight;
        }
        if front.dot(&back).abs() >= 1.0 - EPSILON {
            return CurveCategory::CurvedParallel;
        }
        CurveCategory::Curved
    }

    /// Finds distance along the curve for given time (factor).
    pub fn find_distance_for_time(&self, factor: Scalar) -> Scalar {
        self.split_uninitialized(factor)
            .map(|(mut curve, _)| {
                curve.update_cache();
                curve.length
            })
            .unwrap_or_else(|_| if factor < 0.5 { 0.0 } else { self.length })
    }

    /// Finds time (factor) for given distance along the curve.
    pub fn find_time_for_distance(&self, mut distance: Scalar) -> Scalar {
        if self.length < EPSILON {
            return 0.0;
        }
        distance = distance.max(0.0).min(self.length);
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
        axis_value = axis_value.max(min).min(max);
        let guess = (axis_value - min) / (max - min);
        Some(self.find_time_for(
            Some(guess),
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

impl<T> PartialEq for Curve<T>
where
    T: Clone + Curved + CurvedChange + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.from == other.from
            && self.from_param == other.from_param
            && self.to_param == other.to_param
            && self.to == other.to
            && self.length == other.length
    }
}

fn ray_intersection<T: Curved + CurvedChange>(
    a_point: &T,
    a_direction: &T,
    b_point: &T,
    b_direction: &T,
    guide: Option<&T>,
) -> Option<T> {
    let normal = a_direction.perpendicular(guide)?;
    let a = a_point.dot(&normal);
    let b = b_point.dot(&normal);
    let c = a_direction.dot(&normal);
    let d = b_direction.dot(&normal);
    if (d - c).abs() < EPSILON {
        return None;
    }
    let factor = (a - b) / (d - c);
    Some(b_point.offset(&b_direction.scale(factor)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::factor_iter;

    #[test]
    fn test_curve() {
        let curve = Curve::linear((0.0, 0.0), (10.0, 100.0)).unwrap();

        assert!(curve.sample(0.0).is_nearly_equal_to(&(0.0, 0.0), 1.0e-4));
        assert!(curve.sample(0.5).is_nearly_equal_to(&(5.0, 50.0), 1.0e-4));
        assert!(curve.sample(1.0).is_nearly_equal_to(&(10.0, 100.0), 1.0e-4));

        for factor in factor_iter(10) {
            let provided = curve.find_time_for_axis(factor * 100.0, 1).unwrap();
            let expected = factor;
            assert!(
                provided.is_nearly_equal_to(&expected, 1.0e-4),
                "provided: {:?}, expected: {:?}",
                provided,
                expected
            );
        }

        for factor in factor_iter(10) {
            let provided = curve.sample_along_axis(factor * 100.0, 1).unwrap();
            let expected = (factor * 10.0, factor * 100.0);
            assert!(
                provided.is_nearly_equal_to(&expected, 1.0e-4),
                "provided: {:?}, expected: {:?}",
                provided,
                expected
            );
        }
    }

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
    fn test_curve_split() {
        let curve = Curve::linear((0.0, 0.0), (10.0, 0.0)).unwrap();
        assert!(curve.split(0.0).is_err());
        assert!(curve.split(1.0).is_err());

        let (left, right) = curve.split(0.5).unwrap();
        assert_eq!(left.from, (0.0, 0.0));
        assert_eq!(left.to, (5.0, 0.0));
        assert_eq!(right.from, (5.0, 0.0));
        assert_eq!(right.to, (10.0, 0.0));
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
    fn test_curve_planar() {
        fn project_point_on_plane<T: Curved + CurvedChange>(
            point: &T,
            plane_origin: &T,
            plane_normal: &T,
        ) -> T {
            let plane_normal = plane_normal.normalize();
            let v = plane_origin.delta(point);
            let distance = v.dot(&plane_normal);
            plane_normal.scale(distance).delta(point)
        }

        let curve = Curve::bezier(
            (-100.0, -100.0),
            (0.0, -100.0),
            (0.0, 100.0),
            (100.0, 100.0),
        )
        .unwrap();
        let plane_origin = (0.0, 0.0);
        let plane_normal = (0.0, 0.0);
        let planar = curve.to_planar(&plane_origin, &plane_normal).unwrap();
        assert!(planar.from().is_nearly_equal_to(&(-100.0, -100.0), 1.0e-6));
        assert!(
            planar
                .from_param()
                .is_nearly_equal_to(&(0.0, -100.0), 1.0e-6)
        );
        assert!(planar.to_param().is_nearly_equal_to(&(0.0, 100.0), 1.0e-6));
        assert!(planar.to().is_nearly_equal_to(&(100.0, 100.0), 1.0e-6));
        for factor in factor_iter(10) {
            let a = project_point_on_plane(&curve.sample(factor), &plane_origin, &plane_normal);
            let b = planar.sample(factor);
            assert!(a.is_nearly_equal_to(&b, 1.0e-4));
        }

        let curve = Curve::bezier(
            (-100.0, -100.0, -100.0),
            (0.0, -100.0, -100.0),
            (0.0, 100.0, 100.0),
            (100.0, 100.0, 100.0),
        )
        .unwrap();
        let plane_origin = (0.0, 0.0, 0.0);
        let plane_normal = (0.0, 0.0, 1.0);
        let planar = curve.to_planar(&plane_origin, &plane_normal).unwrap();
        assert!(
            planar
                .from()
                .is_nearly_equal_to(&(-100.0, -100.0, 0.0), 1.0e-6)
        );
        assert!(
            planar
                .from_param()
                .is_nearly_equal_to(&(0.0, -100.0, 0.0), 1.0e-6)
        );
        assert!(
            planar
                .to_param()
                .is_nearly_equal_to(&(0.0, 100.0, 0.0), 1.0e-6)
        );
        assert!(planar.to().is_nearly_equal_to(&(100.0, 100.0, 0.0), 1.0e-6));
        for factor in factor_iter(10) {
            let a = project_point_on_plane(&curve.sample(factor), &plane_origin, &plane_normal);
            let b = planar.sample(factor);
            assert!(a.is_nearly_equal_to(&b, 1.0e-4));
        }

        let curve = Curve::bezier(
            (-100.0, -100.0, -100.0),
            (0.0, -100.0, -100.0),
            (0.0, 100.0, 100.0),
            (100.0, 100.0, 100.0),
        )
        .unwrap();
        let plane_origin = (0.0, 0.0, 0.0);
        let plane_normal = (-1.0, 1.0, 0.0);
        let planar = curve.to_planar(&plane_origin, &plane_normal).unwrap();
        assert!(
            planar
                .from()
                .is_nearly_equal_to(&(-100.0, -100.0, -100.0), 1.0e-6)
        );
        assert!(
            planar
                .from_param()
                .is_nearly_equal_to(&(-50.0, -50.0, -100.0), 1.0e-6)
        );
        assert!(
            planar
                .to_param()
                .is_nearly_equal_to(&(50.0, 50.0, 100.0), 1.0e-6)
        );
        assert!(
            planar
                .to()
                .is_nearly_equal_to(&(100.0, 100.0, 100.0), 1.0e-6)
        );
        for factor in factor_iter(10) {
            let a = project_point_on_plane(&curve.sample(factor), &plane_origin, &plane_normal);
            let b = planar.sample(factor);
            assert!(a.is_nearly_equal_to(&b, 1.0e-4));
        }
    }

    #[test]
    fn test_curve_offset() {
        const DISTANCE: Scalar = 10.0;

        let curve = Curve::bezier((0.0, 0.0), (50.0, 0.0), (100.0, 50.0), (100.0, 100.0)).unwrap();
        assert!(curve.find_extremities().is_empty());
        let offsetted = curve.offset(DISTANCE, None).unwrap();
        assert_eq!(offsetted.len(), 1);
        for factor in factor_iter(10) {
            let a = curve.sample(factor);
            let b = offsetted[0].sample(factor);
            let difference = a.delta(&b).length();
            assert!(
                difference.is_nearly_equal_to(&10.0, 1.0),
                "difference: {} at factor: {}",
                difference,
                factor
            );
        }

        let curve = Curve::bezier((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)).unwrap();
        assert_eq!(curve.find_extremities().len(), 1);
        let offsetted = curve.offset(DISTANCE, None).unwrap();
        assert_eq!(offsetted.len(), 2);
        for factor in factor_iter(5) {
            let a = curve.sample(factor * 0.5);
            let b = offsetted[0].sample(factor);
            let difference = a.delta(&b).length();
            assert!(
                difference.is_nearly_equal_to(&10.0, 1.0),
                "difference: {} at factor: {}",
                difference,
                factor * 0.5
            );
        }
        for factor in factor_iter(5) {
            let a = curve.sample(factor * 0.5 + 0.5);
            let b = offsetted[1].sample(factor);
            let difference = a.delta(&b).length();
            assert!(
                difference.is_nearly_equal_to(&10.0, 1.0),
                "difference: {} at factor: {}",
                difference,
                factor * 0.5 + 0.5
            );
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

    #[test]
    fn test_curve_intersections() {
        let a = Curve::bezier((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)).unwrap();
        let b = Curve::bezier((50.0, 50.0), (0.0, 50.0), (0.0, 0.0), (50.0, 0.0)).unwrap();

        assert_eq!(a.aabb(), (&(0.0, 0.0), &(75.0, 100.0)));
        assert_eq!(b.aabb(), (&(12.5, 0.0), &(50.0, 50.0)));

        #[cfg(not(feature = "scalar64"))]
        let expected = vec![(0.055908203, 0.91918945), (0.055908203, 0.91967773)];
        #[cfg(feature = "scalar64")]
        let expected = vec![
            (0.055908203125, 0.919189453125),
            (0.055908203125, 0.919677734375),
        ];
        assert_eq!(a.find_intersections(&b, 10, 0.1).unwrap(), expected,);
    }

    #[test]
    fn test_curve_category() {
        let curve = Curve::linear((0.0, 0.0), (100.0, 100.0)).unwrap();
        assert_eq!(curve.categorize(), CurveCategory::Straight);

        let curve = Curve::bezier((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)).unwrap();
        assert_eq!(curve.categorize(), CurveCategory::CurvedParallel);

        let curve = Curve::bezier((0.0, 0.0), (50.0, 0.0), (100.0, 50.0), (100.0, 100.0)).unwrap();
        assert_eq!(curve.categorize(), CurveCategory::Curved);
    }

    #[test]
    fn test_ray_intersection() {
        let provided =
            ray_intersection(&(0.0, 0.0), &(1.0, 0.0), &(100.0, 100.0), &(0.0, 1.0), None).unwrap();
        let expected = (100.0, 0.0);
        assert!(
            provided.is_nearly_equal_to(&expected, 1.0e-4),
            "provided: {:?}, expected: {:?}",
            provided,
            expected
        );

        let provided = ray_intersection(
            &(0.0, 0.0, 0.0),
            &(1.0, 0.0, 0.0),
            &(100.0, 100.0, 100.0),
            &(0.0, 1.0, 0.0),
            Some(&(0.0, 0.0, 1.0)),
        )
        .unwrap();
        let expected = (100.0, 0.0, 100.0);
        assert!(
            provided.is_nearly_equal_to(&expected, 1.0e-4),
            "provided: {:?}, expected: {:?}",
            provided,
            expected
        );

        let provided = ray_intersection(
            &(0.0, 0.0),
            &(1.0, 1.0).normalize(),
            &(100.0, 0.0),
            &(-1.0, 1.0).normalize(),
            None,
        )
        .unwrap();
        let expected = (50.0, 50.0);
        assert!(
            provided.is_nearly_equal_to(&expected, 1.0e-4),
            "provided: {:?}, expected: {:?}",
            provided,
            expected
        );

        let provided =
            ray_intersection(&(0.0, 0.0), &(0.0, 1.0), &(100.0, 100.0), &(0.0, 1.0), None);
        assert!(provided.is_none(), "provided: {:?}", provided);
    }
}
