use crate::{
    spline::{Spline, SplineError, SplinePoint, SplinePointDirection},
    utils::range_iter,
    Scalar,
};
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, ops::Range};

/// Easing type.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ease {
    InSine,
    OutSine,
    InOutSine,
    InQuad,
    OutQuad,
    InOutQuad,
    InCubic,
    OutCubic,
    InOutCubic,
    InQuart,
    OutQuart,
    InOutQuart,
    InQuint,
    OutQuint,
    InOutQuint,
    InExpo,
    OutExpo,
    InOutExpo,
    InCirc,
    OutCirc,
    InOutCirc,
    InBack,
    OutBack,
    InOutBack,
}

impl Ease {
    pub fn bezier(self) -> (Scalar, Scalar, Scalar, Scalar) {
        match self {
            Self::InSine => (0.47, 0.0, 0.745, 0.715),
            Self::OutSine => (0.39, 0.575, 0.565, 1.0),
            Self::InOutSine => (0.445, 0.05, 0.55, 0.95),
            Self::InQuad => (0.55, 0.085, 0.68, 0.53),
            Self::OutQuad => (0.25, 0.46, 0.45, 0.94),
            Self::InOutQuad => (0.455, 0.03, 0.515, 0.955),
            Self::InCubic => (0.55, 0.055, 0.675, 0.19),
            Self::OutCubic => (0.215, 0.61, 0.355, 1.0),
            Self::InOutCubic => (0.645, 0.045, 0.355, 1.0),
            Self::InQuart => (0.895, 0.03, 0.685, 0.22),
            Self::OutQuart => (0.165, 0.84, 0.44, 1.0),
            Self::InOutQuart => (0.77, 0.0, 0.175, 1.0),
            Self::InQuint => (0.755, 0.05, 0.855, 0.06),
            Self::OutQuint => (0.23, 1.0, 0.32, 1.0),
            Self::InOutQuint => (0.86, 0.0, 0.07, 1.0),
            Self::InExpo => (0.95, 0.05, 0.795, 0.035),
            Self::OutExpo => (0.19, 1.0, 0.22, 1.0),
            Self::InOutExpo => (1.0, 0.0, 0.0, 1.0),
            Self::InCirc => (0.6, 0.04, 0.98, 0.335),
            Self::OutCirc => (0.075, 0.82, 0.165, 1.0),
            Self::InOutCirc => (0.785, 0.135, 0.15, 0.86),
            Self::InBack => (0.6, -0.28, 0.735, 0.045),
            Self::OutBack => (0.175, 0.885, 0.32, 1.275),
            Self::InOutBack => (0.68, -0.55, 0.265, 1.55),
        }
    }
}

/// Serialzied form of animation phase.
pub type PhaseDef = Vec<SplinePoint<(Scalar, Scalar)>>;

/// Animation phase.
///
/// It focuses on using spline X axis as time and Y axis as values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "PhaseDef")]
#[serde(into = "PhaseDef")]
pub struct Phase {
    spline: Spline<(Scalar, Scalar)>,
    time_frame: Range<Scalar>,
}

impl Default for Phase {
    fn default() -> Self {
        Self::point(1.0).expect("Could not create default point phase")
    }
}

impl Phase {
    /// Builds new animation phase out of spline points.
    pub fn new(mut points: Vec<SplinePoint<(Scalar, Scalar)>>) -> Result<Self, SplineError> {
        let mut time_frame = Scalar::INFINITY..Scalar::NEG_INFINITY;
        for point in &mut points {
            match &mut point.direction {
                SplinePointDirection::Single(dir) => dir.0 = dir.0.max(0.0),
                SplinePointDirection::InOut(from, to) => {
                    from.0 = from.0.min(0.0);
                    to.0 = to.0.max(0.0);
                }
            }
            point.point.0 = point.point.0.max(time_frame.end);
            time_frame.start = time_frame.start.min(point.point.0);
            time_frame.end = time_frame.start.max(point.point.0);
        }
        Ok(Self {
            spline: Spline::new(points)?,
            time_frame,
        })
    }

    /// Builds linear animation phase.
    pub fn linear(
        value_frame: Range<Scalar>,
        mut time_frame: Range<Scalar>,
    ) -> Result<Self, SplineError> {
        if time_frame.start > time_frame.end {
            time_frame = time_frame.end..value_frame.start;
        }
        Self::new(vec![
            SplinePoint::point((time_frame.start, value_frame.start)),
            SplinePoint::point((time_frame.end, value_frame.end)),
        ])
    }

    /// Builds bezier animation phase.
    pub fn bezier(
        (mut x1, mut y1, mut x2, mut y2): (Scalar, Scalar, Scalar, Scalar),
        value_frame: Range<Scalar>,
        mut time_frame: Range<Scalar>,
    ) -> Result<Self, SplineError> {
        if time_frame.start > time_frame.end {
            time_frame = time_frame.end..value_frame.start;
        }
        let distance = (value_frame.end - value_frame.start).abs();
        let duration = time_frame.end - time_frame.start;
        x1 *= duration;
        y1 *= distance;
        x2 = (1.0 - x2) * -duration;
        y2 = (1.0 - y2) * -distance;
        Self::new(vec![
            SplinePoint::new(
                (time_frame.start, value_frame.start),
                SplinePointDirection::Single((x1, y1)),
            ),
            SplinePoint::new(
                (time_frame.end, value_frame.end),
                SplinePointDirection::Single((x2, y2)),
            ),
        ])
    }

    /// Builds easing animation phase.
    pub fn ease(
        ease: Ease,
        value_frame: Range<Scalar>,
        time_frame: Range<Scalar>,
    ) -> Result<Self, SplineError> {
        Self::bezier(ease.bezier(), value_frame, time_frame)
    }

    /// Builds point animation phase.
    pub fn point(point: Scalar) -> Result<Self, SplineError> {
        Self::linear(point..point, 0.0..0.0)
    }

    /// Gets time frame of this phase.
    pub fn time_frame(&self) -> Range<Scalar> {
        self.time_frame.to_owned()
    }

    /// Gets time duration of this phase.
    pub fn duration(&self) -> Scalar {
        self.time_frame.end - self.time_frame.start
    }

    /// Gets slice of spline points.
    pub fn points(&self) -> &[SplinePoint<(Scalar, Scalar)>] {
        self.spline.points()
    }

    /// Sets new spline points.
    pub fn set_points(&mut self, points: Vec<SplinePoint<(Scalar, Scalar)>>) {
        if let Ok(result) = Self::new(points) {
            *self = result;
        }
    }

    /// Gets cached spline.
    pub fn spline(&self) -> &Spline<(Scalar, Scalar)> {
        &self.spline
    }

    /// Gets iterator over uniformly distributed timestamps of this phase.
    pub fn time_iter(&self, steps: usize) -> impl Iterator<Item = Scalar> {
        range_iter(steps, self.time_frame.start, self.time_frame.end)
    }

    /// Sample animation phase at given time.
    pub fn sample(&self, mut time: Scalar) -> Scalar {
        time = time.max(self.time_frame.start).min(self.time_frame.end);
        self.spline.sample_along_axis(time, 0).unwrap().1
    }

    /// Sample animation phase tangent at given time.
    pub fn sample_slope(&self, mut time: Scalar) -> Scalar {
        time = time.max(self.time_frame.start).min(self.time_frame.end);
        self.spline.sample_tangent_along_axis(time, 0).unwrap().1
    }
}

impl TryFrom<PhaseDef> for Phase {
    type Error = SplineError;

    fn try_from(value: PhaseDef) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Phase> for PhaseDef {
    fn from(v: Phase) -> Self {
        v.spline.into()
    }
}
