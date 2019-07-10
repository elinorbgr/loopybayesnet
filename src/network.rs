use crate::LogProbVector;
use ndarray::{Array, ArrayD, Axis, Dimension, RemoveAxis};

struct Node {
    parents: Vec<(usize, LogProbVector)>,
    children: Vec<(usize, LogProbVector)>,
    log_probas: ArrayD<f32>,
    evidence: Option<usize>,
    lambda: Option<LogProbVector>,
    pi: Option<LogProbVector>,
}

impl Node {
    fn evidence_vec(&self) -> LogProbVector {
        if let Some(id) = self.evidence {
            LogProbVector::deterministic(self.log_probas.shape()[0], id)
        } else {
            LogProbVector::uniform(self.log_probas.shape()[0])
        }
    }

    fn compute_lambda(&self) -> LogProbVector {
        self.children
            .iter()
            .fold(self.evidence_vec(), |mut curr_ev, &(_, ref lambda)| {
                curr_ev.prod(lambda);
                curr_ev
            })
    }

    fn compute_and_cache_lambda(&mut self) {
        let lambda = self.compute_lambda();
        self.lambda = Some(lambda.clone());
    }

    fn get_or_compute_lambda(&mut self) -> LogProbVector {
        if self.lambda.is_none() {
            self.compute_and_cache_lambda();
        }
        self.lambda.clone().unwrap()
    }

    fn compute_pi(&self) -> LogProbVector {
        let mut pi = self.log_probas.clone();
        for (_, ref pi_msg) in self.parents.iter().rev() {
            pi = crate::math::log_contract(
                pi.view(),
                pi_msg.log_probabilities(),
                Axis(pi.ndim() - 1),
            );
        }
        // sanity check
        assert!(pi.ndim() == 1);
        LogProbVector::from_log_probabilities(pi.into_shape((self.log_probas.shape()[0],)).unwrap())
    }

    fn compute_and_cache_pi(&mut self) {
        let pi = self.compute_pi();
        self.pi = Some(pi.clone());
    }

    fn get_or_compute_pi(&mut self) -> LogProbVector {
        if self.pi.is_none() {
            self.compute_and_cache_pi();
        }
        self.pi.clone().unwrap()
    }
}

/// Representation of a Bayesian Network
///
/// Once built by adding the nodes one by one, you can use it for inference
/// computation on the graph given some evidence.
pub struct BayesNet {
    nodes: Vec<Node>,
}

impl BayesNet {
    /// Create a new empty Bayesian Network
    pub fn new() -> BayesNet {
        BayesNet { nodes: Vec::new() }
    }

    /// Add a new node to the network
    ///
    /// You need to specify the list of its parents, and an array of probabilities representing `p(x | parents)`.
    /// If the parents are `(p1, ... pk)`, the shape of the array should thus be: `(N, N_p1, ... N_pk)`, where
    /// `N` is the number of possible values for the current variables, and `N_pi` is the number of values of
    /// parent `pi`.
    ///
    /// If the node has no parents, the propabilities must be single-dimenstionnal and represents a prior.
    ///
    /// All values of probabilities should be finite, but the probabilities array does not need to be normalized,
    /// as it will be during the construction process.
    pub fn add_node_from_probabilities<D: Dimension + RemoveAxis>(
        &mut self,
        parents: &[usize],
        probabilities: Array<f32, D>,
    ) -> usize {
        self.add_node_from_log_probabilities(parents, probabilities.mapv(f32::ln))
    }

    /// Add a new node to the network from log-probabilities
    ///
    /// Same as `add_node_from_probabilities`, but the input is in the form of log-probabilities, for greated precision.
    ///
    /// All values of log-probas should be strictly smaller than `+inf`. `-inf` is valid and represents a
    /// probability of 0. The probabilities array does not need to be normalized, as it will be during the construction
    /// process. For example, the log-vector `[0.0, -inf]` will represent a vector of probabilities of `[1.0, 0.0]`.
    ///
    /// Log-probabilities are intepreted as computed with the natural logarithm (base e).
    pub fn add_node_from_log_probabilities<D: Dimension + RemoveAxis>(
        &mut self,
        parents: &[usize],
        mut log_probabilities: Array<f32, D>,
    ) -> usize {
        let id = self.nodes.len();
        // sanity checks
        let shape = log_probabilities.shape();
        assert!(
            shape.len() == parents.len() + 1,
            "Dimensions of log_probas array does not match number of parents"
        );
        for (i, (&val, &parent)) in shape.iter().skip(1).zip(parents.iter()).enumerate() {
            let parent_n_val = self.nodes[parent].log_probas.shape()[0];
            if parent_n_val != val {
                panic!("Dimension {} of log_probas array does not match its associated parent number of element: got {} but parent {} has {}.", i+1, val, parent, parent_n_val);
            }
        }

        // the shapes match, proceed to insert the node
        for &p in parents {
            self.nodes[p]
                .children
                .push((id, LogProbVector::uniform(shape[0])));
        }

        crate::math::normalize_log_probas(log_probabilities.view_mut());

        let parents = parents
            .iter()
            .map(|&p| {
                (
                    p,
                    LogProbVector::uniform(self.nodes[p].log_probas.shape()[0]),
                )
            })
            .collect();

        self.nodes.push(Node {
            parents,
            children: Vec::new(),
            log_probas: log_probabilities.into_dyn(),
            evidence: None,
            lambda: None,
            pi: None,
        });

        id
    }

    /// Sets the evidence for the network
    ///
    /// Input is interpreted as a list of `(node_id, node_value)`. Out-of-range evidence is not checked, but
    /// will result into a probability of `0`.
    pub fn set_evidence(&mut self, evidence: &[(usize, usize)]) {
        // Reset the evidences to None before applying the new evidence
        for node in &mut self.nodes {
            node.evidence = None;
        }
        for &(node, value) in evidence {
            self.nodes[node].evidence = Some(value);
        }
    }

    /// Resets the internal state of the inference algorithm, to begin a new inference
    pub fn reset_state(&mut self) {
        for node in &mut self.nodes {
            for &mut (_, ref mut msg) in &mut node.children {
                msg.reset();
            }
            for &mut (_, ref mut msg) in &mut node.parents {
                msg.reset();
            }
            node.lambda = None;
            node.pi = None;
        }
    }

    /// Compute the current state belief of each node according to the current internal messages
    pub fn beliefs(&self) -> Vec<LogProbVector> {
        self.nodes
            .iter()
            .map(|node| {
                let mut lambda = node.lambda.clone().unwrap_or_else(|| node.compute_lambda());
                let pi = node.pi.clone().unwrap_or_else(|| node.compute_pi());
                lambda.prod(&pi);
                lambda.renormalize();
                lambda
            })
            .collect()
    }

    /// Compute one step of the Loopy Belief Propagation Algorithm
    ///
    /// The algorithm can be run for any number of steps. it is up to you to decide when to stop.
    ///
    /// A classic stopping criterion is when the yielded beliefs stop significantly changing.
    pub fn step(&mut self) {
        // At the start of the algorithm, we assume all present cached values for lambda and pi are valid for
        // the currently stored messages. We will then compute the new messages and invalidate the caches.

        // Compute the new messages and store them into thes two big vectors, once this done we will replace
        // them into the graph.
        // Their layout is (from, to, content). We pre-allocate the correct capacity.
        let mut pi_msgs: Vec<(usize, usize, LogProbVector)> =
            Vec::with_capacity(self.nodes.iter().map(|n| n.children.len()).sum());
        let mut lambda_msgs: Vec<(usize, usize, LogProbVector)> =
            Vec::with_capacity(self.nodes.iter().map(|n| n.parents.len()).sum());

        for (id, node) in self.nodes.iter_mut().enumerate() {
            // compute the pi messages:
            let mut pi = node.get_or_compute_pi();
            pi.prod(&node.evidence_vec());
            for &(child_id, _) in &node.children {
                let mut msg = node
                    .children
                    .iter()
                    .filter(|&&(cid, _)| cid != child_id)
                    .fold(pi.clone(), |mut acc, (_, ref v)| {
                        acc.prod(v);
                        acc
                    });
                msg.renormalize();
                pi_msgs.push((id, child_id, msg));
            }

            // compute the lambda messages:
            let lambda = node.get_or_compute_lambda();
            for &(parent_id, _) in &node.parents {
                let acc = node
                    .parents
                    .iter()
                    .enumerate()
                    .rev()
                    .filter(|&(_, &(pid, _))| pid != parent_id)
                    .fold(node.log_probas.clone(), |acc, (axid, &(_, ref v))| {
                        crate::math::log_contract(acc.view(), v.log_probabilities(), Axis(axid + 1))
                    });
                let acc =
                    crate::math::log_contract(acc.view(), lambda.log_probabilities(), Axis(0));
                assert!(acc.ndim() == 1);
                let shape = (acc.len(),);
                let mut msg = LogProbVector::from_log_probabilities(acc.into_shape(shape).unwrap());
                msg.renormalize();
                lambda_msgs.push((id, parent_id, msg));
            }

            // invalidate the cached lambda & pi
            node.lambda = None;
            node.pi = None;
        }

        // Finally, store the msgs in their new place
        for (from, to, msg) in pi_msgs {
            if let Some(&mut (_, ref mut place)) = self.nodes[to]
                .parents
                .iter_mut()
                .find(|&&mut (parent_id, _)| parent_id == from)
            {
                *place = msg;
            } else {
                panic!(
                    "Message from {} to {} who doesn't recognize its parent?!",
                    from, to
                );
            }
        }
        for (from, to, msg) in lambda_msgs {
            if let Some(&mut (_, ref mut place)) = self.nodes[to]
                .children
                .iter_mut()
                .find(|&&mut (child_id, _)| child_id == from)
            {
                *place = msg;
            } else {
                panic!(
                    "Message from {} to {} who doesn't recognize its child?!",
                    from, to
                );
            }
        }
    }
}
