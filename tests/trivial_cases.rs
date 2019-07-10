use loopybayesnet::BayesNet;
use ndarray::{Array1, Array2};

pub fn assert_all_close(a: &Array1<f32>, b: &[f32], eps: f32) {
    if a.len() != b.len() ||  a.iter().zip(b.iter()).any(|(&a, &b)| (a-b).abs() > eps) {
        panic!("{:?} != {:?} (+/- {})", a, b, eps);
    }
}

#[test]
fn two_nodes() {
    let mut net = BayesNet::new();
    let _node1 = net.add_node_from_probabilities(&[], Array1::from(vec![0.5, 0.5]));
    let _node2 = net.add_node_from_probabilities(&[_node1], Array2::from(vec![[0.5, 1.0], [0.5, 0.0]]));

    // no evidence should yield 50/50 marginals
    net.reset_state();
    net.set_evidence(&[]);
    for _ in 1..10 { net.step(); }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[0.5, 0.5], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[0.75, 0.25], 0.001);

    // Positive evidence on node 2 should 100% determine the node 1
    net.reset_state();
    net.set_evidence(&[(1,1)]);
    for _ in 1..10 { net.step(); }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[1.0, 0.0], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[0.0, 1.0], 0.001);

    // Negative evidence on node 2 should not 100% determine node 1
    net.reset_state();
    net.set_evidence(&[(1,0)]);
    for _ in 1..10 { net.step(); }
    let beliefs = net.beliefs();
    assert_all_close(&beliefs[0].as_probabilities(), &[0.333, 0.666], 0.001);
    assert_all_close(&beliefs[1].as_probabilities(), &[1.0, 0.0], 0.001);
}