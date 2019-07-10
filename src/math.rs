use ndarray::{Array, ArrayView, ArrayView1, ArrayViewMut, Axis, Dimension, RemoveAxis};

pub fn log_sum_exp_vec(x: ArrayView1<f32>) -> f32 {
    let max_log = x.fold(std::f32::NEG_INFINITY, |old_max, &v| f32::max(old_max, v));
    if !max_log.is_finite() {
        // if max_log is +inf, result will be +inf anyway
        // if max_log is -inf, then all log values are -inf, and the result of the log_sum_exp is too
        max_log
    } else {
        max_log + x.mapv(|v| (v - max_log).exp()).sum().ln()
    }
}

pub fn log_sum_exp<D: Dimension + RemoveAxis>(
    x: ArrayView<f32, D>,
    axis: Axis,
) -> Array<f32, D::Smaller> {
    x.map_axis(axis, log_sum_exp_vec)
}

pub fn log_sum_exp_keepdim<D: Dimension + RemoveAxis>(
    x: ArrayView<f32, D>,
    axis: Axis,
) -> Array<f32, <D::Smaller as Dimension>::Larger> {
    log_sum_exp(x, axis).insert_axis(axis)
}

pub fn log_contract<D: Dimension + RemoveAxis>(
    tensor: ArrayView<f32, D>,
    vector: ArrayView1<f32>,
    axis: Axis,
) -> Array<f32, D::Smaller> {
    tensor.map_axis(axis, |v| {
        let mut v = v.into_owned();
        v += &vector;
        log_sum_exp_vec(v.view())
    })
}

pub fn normalize_log_probas<D: Dimension + RemoveAxis>(mut x: ArrayViewMut<f32, D>) {
    let lsm = log_sum_exp_keepdim(x.view(), Axis(0));
    x -= &lsm;
}
