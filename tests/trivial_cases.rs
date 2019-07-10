use loopybayesnet::BayesNet;
use ndarray::{Array1, Array2, Array3};

pub fn assert_all_close(a: &Array1<f32>, b: &[f32], eps: f32) {
    if a.len() != b.len() || a.iter().zip(b.iter()).any(|(&a, &b)| (a - b).abs() > eps) {
        panic!(
            "{:?} != {:?} (+/- {})",
            a.view().into_slice().unwrap(),
            b,
            eps
        );
    }
}

#[test]
fn two_nodes() {
    let mut net = BayesNet::new();
    let _node1 = net.add_node_from_probabilities(&[], Array1::from(vec![0.5, 0.5]));
    let _node2 =
        net.add_node_from_probabilities(&[_node1], Array2::from(vec![[0.5, 1.0], [0.5, 0.0]]));

    // no evidence should yield 50/50 marginals
    net.reset_state();
    net.set_evidence(&[]);
    for _ in 1..10 {
        net.step();
    }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[0.5, 0.5], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[0.75, 0.25], 0.001);

    // Positive evidence on node 2 should 100% determine the node 1
    net.reset_state();
    net.set_evidence(&[(1, 1)]);
    for _ in 1..10 {
        net.step();
    }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[1.0, 0.0], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[0.0, 1.0], 0.001);

    // Negative evidence on node 2 should not 100% determine node 1
    net.reset_state();
    net.set_evidence(&[(1, 0)]);
    for _ in 1..10 {
        net.step();
    }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[0.333, 0.666], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[1.0, 0.0], 0.001);
}

#[test]
fn multi_valued() {
    let mut net = BayesNet::new();
    let _node1 = net.add_node_from_probabilities(&[], Array1::from(vec![0.5, 0.4, 0.1]));
    let _node2 = net.add_node_from_probabilities(
        &[_node1],
        Array2::from(vec![[0.8, 0.2, 1.0], [0.2, 0.8, 0.0]]),
    );
    let _node3 = net.add_node_from_probabilities(
        &[_node1, _node2],
        Array3::from(vec![
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ]),
    );

    // no evidence
    net.reset_state();
    net.set_evidence(&[]);
    for _ in 1..3 {
        net.step();
    }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[0.5, 0.4, 0.1], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[0.58, 0.42], 0.001);
    // these are not the actual probabilities, this is a case where the approximation is wrong
    assert_all_close(
        &beliefs[2].as_probabilities(),
        &[0.232, 0.458, 0.21, 0.1],
        0.001,
    );
}
