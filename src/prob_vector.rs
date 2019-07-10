use ndarray::{Array1, ArrayView1};

/// The representation of a probability vector in log-space
///
/// Log-space manipulation of probabilitys is stabler regarding vectors
/// with values very close to 0 or 1. This uses the natural logarithm (base `e`).
///
/// The content of this log-proba vector may not be normalized: adding a constant
/// value to all entries of the vector does not change the normalized probability
/// it represents.
#[derive(Debug, Clone)]
pub struct LogProbVector {
    log_probs: Array1<f32>,
}

impl LogProbVector {
    /// Create an unnormalized log-probability vector representing an uniform distribution
    pub fn uniform(n: usize) -> LogProbVector {
        LogProbVector {
            log_probs: vec![0.0; n].into(),
        }
    }

    /// Create an unnormalized log-probability vector representing a deterministic distribution
    /// choosing value `i` from the `n` possible
    ///
    /// If `i >= n`, this returns a vector assigning 0 probability to every value.
    pub fn deterministic(n: usize, i: usize) -> LogProbVector {
        let mut data = vec![std::f32::NEG_INFINITY; n];
        if i < n {
            data[i] = 0.0;
        }
        LogProbVector {
            log_probs: data.into(),
        }
    }

    /// Wrap an array of log-probabilities into a log-probability vector
    pub fn from_log_probas(log_probs: Array1<f32>) -> LogProbVector {
        LogProbVector { log_probs }
    }

    /// Access the underlying array of log-probas
    pub fn log_probas(&self) -> ArrayView1<f32> {
        self.log_probs.view()
    }

    /// Get the normalized probabilities represented by this log-probability vector
    pub fn as_proba(&self) -> Array1<f32> {
        let probs = self.log_probs.mapv(f32::exp);
        let norm_cst = probs.sum();
        if norm_cst > 0.0 {
            probs / norm_cst
        } else {
            // probs are all 0
            probs
        }
    }

    /// Renormalize the log-probability vector so that its content represent exactly the log
    /// of a normalized probability distribution.
    pub fn renormalize(&mut self) {
        let sum = crate::math::log_sum_exp_vec(self.log_probs.view());
        self.log_probs.map_inplace(|v| *v -= sum);
    }

    /// Multiply the given log-probability vector into this one.
    ///
    /// NB: Multiplication is done in probability space, hence the log-probabilities are *summed*
    /// As a result, the log-probability vector will no longer be normalized if it was.
    pub fn prod(&mut self, other: &LogProbVector) {
        self.log_probs += &other.log_probs;
    }

    /// Resets this log-probas vector to a uniform distribution
    pub fn reset(&mut self) {
        for v in self.log_probs.iter_mut() {
            *v = 0.0;
        }
    }
}
