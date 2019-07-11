use loopybayesnet::BayesNet;
use ndarray::{Array1, Array2, Array3};

// Let us here create a Bayesian Network that we will use to weight evidence about whether the Earth
// is flat or spherical.
//
// I'm not an astraunot, and I haven't been to space to see for myself the shape of the Earth. I can only
// try to infer it from the partial evidence I can see for myself.
//
// Using Bayesian probabilities, we can model the relationship between these evidences and how they
// support the various hypothsis. In this context, probabilities have nothing to do with randomness, but
// they represent how much we believe that a prioposition is plausible. A plausibility very near 0
// means we really don't believe it, and a plausibility near 1 means we really believe it.
//
// However, we also need to take into account two things:
//
// - We cannot evaluate the plausibility of an hypothesis alsone, but only compared to other hypotheses
// - Our mind tends to work in logarithmic space, so to compare two hypothese H1 and H2, we should
//   actually look at log(p(H1) / p(H2)), with this small example of possible values
//   (using base 10 logarithm):
//
//   *  5 : I'm extremely confident that H1 is much more plausible than H2
//   *  3 : I think H1 is more plausible than H2
//   *  1 : H1 seems slightly more pausible than H2
//   *  0 : I cannot decide between H1 and H2
//   * -1 : H2 seems slightly more plausible than H1
//   * -3 : I think H2 is more plausible than H1
//   * -5 : I'm extremely confident that H2 is much more plausible than H1
//
// To compare several hypotheses, we'll thus work directly in log-space (which loopybayesnet already does
// for numerical stability). However, loopybayesnet works with natural logarithms, so we'll need to
// remember to multiply or divide our values by ln(10) when appropriate.

fn main() {
    let mut net = BayesNet::new();
    let log10 = 10f32.ln();

    // With all that said, lets start our modelisation. First fo all, there is the main hypothesis we want to
    // determine: is the Earth round or flat? We'll create a node to represent this. Let's assign the following
    // values: 0 = the Earth is round, 1 = the Earth is flat. Assuming no evidence at all, we have no reason
    // to prefer one or the other, so we put an uniform prior on this node:
    //
    // Again, remember that the important values is the difference between log P(H1) and log P(H2): adding a
    // constant value to both does not change anything.
    let flat = net.add_node_from_log_probabilities(&[], Array1::from(vec![0.0, 0.0]));

    // Now then, an argument often raised is that the Earth is flat and that there is some conspiracy to make
    // us believe that it is in fact round. We shall not dismiss this argument without considering it, and thus
    // it deserves a node in our graph.
    //
    // To define this node, we'll consider the plausibility of the existence of this conspiracy depending on
    // whether we suppose that the Earth is round or flat.
    //
    // If the Earth is round, this conspiracy has no reason for existing, so P(conspiracy | round) will be
    // very close to 0. We'll take log P(conspiracy | round) / P(not conspiracy | round) = -5 to reflect that.
    //
    // If the Earth is flat, this conspiracy may exist, even though we are not clear about what its motivations
    // would be. So, lets take P(conspiracy | flat) / P(not conspiracy | flat) = -2. This seems unlikely, but
    // why not after all.
    let conspiracy = net.add_node_from_log_probabilities(
        &[flat],
        Array2::from(
            vec![[ 0.0,  0.0],  // these are the log-probabilities of "not-conspiracy", we leave them to 0 as
                                // only the difference matters
                 [-5.0, -2.0]]  // these are the log-probabilities of "conspiracy", as we chose them earlier
        ) * log10 // multiply the values by log(10) to bring them back into base e
    );

    // With that in place, lets look at the actual evidence we see.

    // The first, most obvious one, is that the Earth *looks* flat from the ground.
    // If the Earth really is flat, this is quite natural, so we would expect the Earth to look flat:
    //     log P(looks flat | flat) / P(not looks flat | flat) = 5
    // If the Earth is round, we are told it is still very very large, so it is not very suprizing
    // that it looks flat at our scale:
    //     log P(looks flat | round) / P(not looks flat | round) = 3
    let looks_flat = net.add_node_from_log_probabilities(
        &[flat],
        Array2::from(
            vec![[ 0.0, 0.0],
                 [ 3.0, 5.0]]
        ) * log10
    );

    // A second evidence we observe, is the existence of the horizon, and the fact that objects can disappear
    // behind it.
    //
    // If the Earth is round, this is perfectly natural from a geometric point of view:
    //     log P(horizon | round) / P(not horizon | round) = 5
    // If the Earth is flat, we do not have a clear justification of *why* the horizon exist, but we neither
    // have a clear evidence of why it should not exist. There may be some particular optical phenomenon due
    // to temperature differences in the air, just like mirages in the desert. So lets remain conservative:
    //     log P(horizon | flat) / P(not horizon | flat) = 0
    let horizon = net.add_node_from_log_probabilities(
        &[flat],
        Array2::from(
            vec![[ 0.0, 0.0],
                 [ 5.0, 0.0]]
        ) * log10
    );

    // Third evidence, all the photos we got of the Earth from space, on which it seems round.
    //
    // If the Earth is round, it's not suprising that it looks round on these photos, though we know they are
    // often photoshopped to be more visually appealing:
    //     log P(photos | round) / P(not photos | round) = 4
    // If the Earth is flat though, it depends on whether there is a conspiracy:
    //  - if there is no conspiracy, it's quite suprising that the Earth would look round on these photos, but
    //    again small photoshopping could unvoluntarily give false impressions:
    //        log P(photos | flat, not conspiracy) / P(not photos | flat, not conspiracy) = -4
    //  - if there is a conspiracy, then it's obvious that the photos would show a round Earth, as it is the
    //    exact goal of this conspiracy!
    //        log P(photos | flat, conspiracy) / P(not photos | flat, conspiracy) = 5
    let photos = net.add_node_from_log_probabilities(
        &[flat, conspiracy],
        Array3::from(
            vec![[[0.0, 0.0], [ 0.0, 0.0]], // innermost array is "conspiracy / not conspiracy", second array
                 [[4.0, 4.0], [-4.0, 5.0]]] // is "flat / round". If the Earth is round, the presence of the
                                            // conspiracy is irrelevant.
        ) * log10
    );

    // Fourth evidence: we never had any credible leak about the existence of the conspiracy.
    //
    // If there is no conspiracy, this is quite natural that we never saw any leak, though we can imagine
    // seeing a leak of a non-existent conspiracy.
    //    log P(leak | not conspiracy) / P(not leak | not conspiracy) = -4
    // If there is a conspiracy, it must have been running for quite a long time, given many people have believed
    // the Earth round for hundred of years. However, we have reasonable evidence showing that big conspiracies
    // tend to be relatively quickly leaked, possibly unvoluntarily. So if there is such a conspiracy, we should
    // expect to see at least some leaks.
    //    log P(leak | conspiracy) / P(not leak | not conspiracy) = 3
    let leak = net.add_node_from_log_probabilities(
        &[conspiracy],
        Array2::from(
            vec![[ 0.0, 0.0],
                 [-4.0, 3.0]]
        ) * log10
    );


    //
    // Now that we have finished our model, it's actually time to run the network
    //

    // First, set the evidence we actually have:
    net.set_evidence(&[
        (looks_flat, 1), // The Earth does look flat
        (horizon, 1),    // We see an horizon
        (photos, 1),     // We see photos from space
        (leak, 0),       // We haven't seen any leak
    ]);

    // Now, lets run the algorithm:
    net.reset_state();
    for _ in 0..20 {
        net.step();
    }

    // Finnaly, lets find out the results. Run this program to get the verdict ;)
    let beliefs = net.beliefs();

    println!("log Evidence ratios (5 = very in favor, 0 = indecisive, -5 = very not in favor):");

    let log_ratios = beliefs[flat].log_probabilities();
    println!(" - flat Earth: {}", (log_ratios[1] - log_ratios[0]) / log10);

    let log_ratios = beliefs[conspiracy].log_probabilities();
    println!(" - conspiracy: {}", (log_ratios[1] - log_ratios[0]) / log10);
}