use loopybayesnet::BayesNet;

fn main() {
    let mut net = BayesNet::new();

    // create a small graph from the classic example:
    //
    // +----------+          +----------------------+
    // | It rains | -------> | Sprinkler is running |
    // +----------+          +----------------------+
    //       |                 |
    //       +----+     +------+
    //            |     |
    //            v     v
    //        +--------------+
    //        | Grass is Wet |
    //        +--------------+

    // Rain has no parents, it is a prior
    // We have P(not Rain) = 0.8, P(Rain) = 0.8
    let rain = net.add_node(
        &[],
        ndarray::Array1::from(vec![0.8, 0.2]).mapv(f32::ln)
    );

    // Sprinkler has a parent (Rain)
    // We have P(not Sprinkler | not Rain) = 0.60, P(not Sprinkler | Rain) = 0.99
    //         P(    Sprinkler | not Rain) = 0.40, P(    Sprinkler | Rain) = 0.01
    let sprinkler = net.add_node(
        &[rain],
        ndarray::Array2::from(vec![[0.60, 0.99],
                                  [0.40, 0.01]]).mapv(f32::ln)
    );

    // Wet has 2 parents (Rain and Sprinkler)
    // We have P(not Wet | not Rain, not Sprinkler) = 1.0, P(not Wet | not Rain, Sprinkler) = 0.1
    //         P(not Wet | Rain,     not Sprinkler) = 0.2, P(not Wet | Rain,     Sprinkler) = 0.01
    //         P(    Wet | not Rain, not Sprinkler) = 0.0, P(    Wet | not Rain, Sprinkler) = 0.9
    //         P(    Wet | Rain,     not Sprinkler) = 0.8, P(    Wet | Rain,     Sprinkler) = 0.99
    let wet = net.add_node(
        &[rain, sprinkler],
        ndarray::Array3::from(vec![[[1.0, 0.1], [0.2, 0.01]],
                                   [[0.0, 0.9], [0.8, 0.99]]]).mapv(f32::ln)
    );

    // We can now do some inferences
    // First, compute the marginal probabilities of the network without any evidence
    net.reset_state();
    net.set_evidence(&[]);
    println!("===== raw marginal probabilities =====");
    for _ in 1..10 {
        // this net converges pretty quickly
        net.step();
    }
    let beliefs = net.beliefs();
    println!("    Rain: {}", beliefs[rain].as_proba());
    println!("    Sprinkler: {}", beliefs[sprinkler].as_proba());
    println!("    Wet: {}", beliefs[wet].as_proba());

    println!();
    println!("===== marginal probabilities assuming the grass is wet =====");
    // Now, assuming we see the grass we, what can we infer from this ?
    net.reset_state();
    net.set_evidence(&[(wet, 1)]);
    for _ in 1..50 {
        // this net is slower to converge
        net.step();
    }
    let beliefs = net.beliefs();
    println!("    Rain: {}", beliefs[rain].as_proba());
    println!("    Sprinkler: {}", beliefs[sprinkler].as_proba());
    println!("    Wet: {}", beliefs[wet].as_proba());

    println!();
    println!("===== marginal probabilities assuming the sprinkler is running =====");
    // Evidence doesn't need to be at the last node
    net.reset_state();
    net.set_evidence(&[(sprinkler, 1)]);
    for _ in 1..10 {
        // this one is quick to converge too
        net.step();
    }
    let beliefs = net.beliefs();
    println!("    Rain: {}", beliefs[rain].as_proba());
    println!("    Sprinkler: {}", beliefs[sprinkler].as_proba());
    println!("    Wet: {}", beliefs[wet].as_proba());

    println!();
    println!("===== marginal probabilities assuming it rains =====");
    // Evidence can even be at the prior !
    net.reset_state();
    net.set_evidence(&[(rain, 1)]);
    for _ in 1..10 {
        // this one is quick to converge too
        net.step();
    }
    let beliefs = net.beliefs();
    println!("    Rain: {}", beliefs[rain].as_proba());
    println!("    Sprinkler: {}", beliefs[sprinkler].as_proba());
    println!("    Wet: {}", beliefs[wet].as_proba());
}