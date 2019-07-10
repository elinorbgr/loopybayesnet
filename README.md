# Loopy Bayes Net

An implementation of the Loopy Belief Propagation algorithm for Bayesian Networks

### Bayesian networks & Loopy belief propagation

Bayesian networks can be used to encode a set of causal or logical probabilistic dependency
between events. They take the shape of directed acyclic graphs, each node being associated
with a probability table defining the probability of it taking each possible values depending
on the values of its parents. For further details, you can check
[Wikipedia](https://en.wikipedia.org/wiki/Bayesian_network).

The Loopy Belief Propagation is an algorithm computes an approximation of the marginal probability
distribution of each node of the network, conditionned to the value of a chosen set of "observed"
variables, for which the values are set beforethand.

This is an approximation, which behaves as if the parents of each node were conditionnaly
independent given the node. This is only true if the considered graph is actually a tree (there
are no undirected loop), in which case the approximation is exact.

A typical failure case of this algorithm is when some nodes have parents that are both strongly
correlated and very random (which is notably the case for the `simple_net` example ;) ). Then,
even if the algorithm converges (it is not always the case), it is likely to converge to a wrong
value.

On the other hand, for networks where the observations almost certainly determine the value of the
rest of the network (which is often the case in real-world problems), the Loopy Belief Propagation
algorithm provides a very good approximation (see [arXiv:1301.6725](https://arxiv.org/pdf/1301.6725.pdf)
for a study about it for example).

### How to use this crate

This crate allows you, given a fully-specified Bayesian Network and a set of observations, to
iteratively run the Loopy Belief Propagation algorithm on it.

```rust
use loopybayesnet::BayesNet;
use ndarray::{Array1, Array2};

let mut net = BayesNet::new();

// First insert your modes to create your network
let node1 = net.insert_node_from_probabilities(
    &[], // This node does not have any parent,
    Array1::from(vec![0.2, 0.3, 0.5]) // This node can take 3 values, and these are the
                                      // associated prior probabilities
);
let node2 = net.insert_node_from_probabilities(
    &[node1], // This node has node1 as a parent
    Array2::from(vec![   // This node can take 2 values
        [0.4, 0.3, 0.7], // these are the probabilities of value 0 depending on the value of node1
        [0.6, 0.7, 0.3], // these are the probabilities of value 1 depending on the value of node1
    ])
);

// Now that the net is set, we can run the algorithm

// Reset the internal state which is kept, as the algorithm is iterative
net.reset_state();
// Set as evidence that node2 = 0
net.set_evidence(&[(node2, 0)]);

for _ in range 0..3 {
    // Iterate over the steps of the algorithms until convergence.
    // As a rule of thumb, the number of necessary iterations is often of the same order as the
    // diameter of the graph, but it may be longer with graphs containing loops.
    net.step();
    // between each step we can observe the current state of the algorithm
    let beliefs = net.beliefs();
    // the algorithm works internally with log-probabilities for numerical stability
    // the `.as_probabilities()` call converts them back to regular normalized probabilities.
    println!("Marginals for node1: {:?", beliefs[node1].as_probabilities());
    println!("Marginals for node2: {:?", beliefs[node2].as_probabilities());
}
```

### Quality of the implementation

The implementation should be mostly correct, given the algorithm is quite clear to follow
and `ndarray` handles the difficult stuff ;)

I made no attempt to optimize the performance though, so it may run slowly on very large
graphs with many values and many parents per node.