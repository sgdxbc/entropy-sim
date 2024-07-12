use std::collections::BTreeMap;

use rand::{rngs::StdRng, seq::IteratorRandom, Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Geometric};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type HashMap<K, V> = rustc_hash::FxHashMap<K, V>;

trait FailureGenerator {
    fn rate(&self, system: &mut System, rng: &mut impl Rng);
}

trait Protocol {
    fn enter(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng);

    fn maintain(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng) -> bool;
}

type Digest = [u8; 32];

struct Node {
    equiv: NodeEquiv,
    fragments: HashMap<Digest, FragmentIndex>,
}

type NodeEquiv = u32;
type FragmentIndex = usize;

struct System {
    step_count: u32,
    failure_count: u32,

    domains: Vec<(Vec<NodeEquiv>, f64)>,
    nodes: BTreeMap<NodeId, Node>,
    node_equiv: Vec<NodeId>,
    data: HashMap<Digest, HashMap<FragmentIndex, NodeId>>,

    config: SystemParameters,
}

type NodeId = [u8; 32];

struct SystemParameters {
    num_node: usize,
    num_data: usize,
}

impl System {
    fn new(config: SystemParameters) -> Self {
        Self {
            config,
            step_count: 0,
            failure_count: 0,
            domains: Default::default(),
            nodes: Default::default(),
            node_equiv: Default::default(),
            data: Default::default(),
        }
    }

    fn init(
        &mut self,
        failure_generator: &impl FailureGenerator,
        protocol: &mut impl Protocol,
        rng: &mut impl Rng,
    ) {
        self.node_equiv
            .resize_with(self.config.num_node, Default::default);
        for equiv in 0..self.config.num_node {
            let id = rng.gen();
            let replaced = self.nodes.insert(
                id,
                Node {
                    equiv: equiv as _,
                    fragments: Default::default(),
                },
            );
            assert!(replaced.is_none());
            self.node_equiv[equiv] = id
        }
        for _ in 0..self.config.num_data {
            let digest = rng.gen();
            let replaced = self.data.insert(digest, Default::default());
            assert!(replaced.is_none());
            protocol.enter(&digest, self, rng)
        }
        failure_generator.rate(self, rng)
    }

    fn finalized(&self) -> bool {
        self.data.is_empty()
    }

    fn step(&mut self, protocol: &mut impl Protocol, rng: &mut impl Rng) {
        for (nodes, failure_rate) in &self.domains {
            if !rng.gen_bool(*failure_rate) {
                continue;
            }
            for equiv in nodes {
                let id = &self.node_equiv[*equiv as usize];
                let Some(mut node) = self.nodes.remove(id) else {
                    continue;
                };
                self.failure_count += 1;
                for (digest, index) in node.fragments.drain() {
                    self.data
                        .get_mut(&digest)
                        .expect("data is not lost (yet)")
                        .remove(&index)
                        .expect("fragment is present");
                }
                let id = rng.gen();
                let replaced = self.nodes.insert(id, node);
                assert!(replaced.is_none());
                self.node_equiv[*equiv as usize] = id
            }
        }
        for digest in self.data.keys().cloned().collect::<Vec<_>>() {
            if protocol.maintain(&digest, self, rng) {
                continue;
            }
            self.data
                .remove(&digest)
                .expect("data is not lost before failed repair");
            self.print()
        }
        self.step_count += 1
    }

    fn store(&mut self, digest: &Digest, index: FragmentIndex, id: &NodeId) {
        let replaced = self
            .nodes
            .get_mut(id)
            .expect("node exists")
            .fragments
            .insert(*digest, index);
        assert!(replaced.is_none());
        let replaced = self
            .data
            .get_mut(digest)
            .expect("data exists")
            .insert(index, *id);
        assert!(replaced.is_none())
    }

    fn garbage_collect(&mut self, digest: &Digest, id: &NodeId) {
        let index = self
            .nodes
            .get_mut(id)
            .expect("node exists")
            .fragments
            .remove(digest)
            .expect("fragment exists");
        self.data
            .get_mut(digest)
            .expect("data exists")
            .remove(&index)
            .expect("fragment exists");
    }

    fn print(&self) {
        let num_step_failure = self.failure_count as f32 / self.step_count as f32;
        let churn = self.failure_count as f32
            / self.config.num_node as f32
            // / (self.step_count as f32 / 1000.)
            / self.step_count as f32
            * 100.;
        let redundancy = self
            .data
            .values()
            .map(|fragments| fragments.len())
            .sum::<usize>() as f32
            / self.data.len() as f32;
        eprint!(
            "{:<120}",
            format!(
                "\r[step {:7}] {:3} alive, {num_step_failure:4.2} failures/step ({churn:4.2}% churn), {redundancy:4.2} fragments/data",
                self.step_count,
                self.data.len(),
            )
        )
    }
}

struct NopProtocol {
    n: usize,
    k: usize,
}

impl Protocol for NopProtocol {
    fn enter(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng) {
        let ids = system.nodes.keys().copied().choose_multiple(rng, self.n);
        assert!(ids.len() == self.n);
        for (index, id) in ids.into_iter().enumerate() {
            system.store(data_digest, index, &id)
        }
    }

    fn maintain(&mut self, digest: &Digest, system: &mut System, _: &mut impl Rng) -> bool {
        system.data[digest].len() >= self.k
    }
}

struct RingSuccessorProtocol {
    n: usize,
    k: usize,
}

impl RingSuccessorProtocol {
    fn store(&self, digest: &Digest, system: &mut System) {
        let mut nodes = system
            .nodes
            .range(*digest..)
            .take(self.n)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        if nodes.len() < self.n {
            nodes.extend(system.nodes.keys().take(self.n - nodes.len()).copied())
        }
        for (index, node_id) in nodes.iter().enumerate() {
            if let Some(other_id) = system.data[digest].get(&index) {
                if other_id != node_id {
                    let other_id = *other_id;
                    system.garbage_collect(digest, &other_id)
                }
            }
        }
        for (index, node_id) in nodes.iter().enumerate() {
            if !system.data[digest].contains_key(&index) {
                system.store(digest, index, node_id)
            }
        }
    }
}

impl Protocol for RingSuccessorProtocol {
    fn enter(&mut self, data_digest: &Digest, system: &mut System, _: &mut impl Rng) {
        self.store(data_digest, system)
    }

    fn maintain(&mut self, data_digest: &Digest, system: &mut System, _: &mut impl Rng) -> bool {
        if system.data[data_digest].len() < self.k {
            return false;
        }
        self.store(data_digest, system);
        true
    }
}

struct IndependentFailure {
    rate: f64,
}

impl FailureGenerator for IndependentFailure {
    fn rate(&self, system: &mut System, _: &mut impl Rng) {
        for equiv in 0..system.config.num_node {
            system.domains.push((vec![equiv as _], self.rate))
        }
        eprintln!("generated independent failures")
    }
}

struct CorrelatedFailure {
    num_domain: usize,
    size_distr: Geometric,
    rate_distr: Exp<f64>,
}

impl FailureGenerator for CorrelatedFailure {
    fn rate(&self, system: &mut System, rng: &mut impl Rng) {
        for i in 0..self.num_domain {
            let size = self.size_distr.sample(rng) + 2;
            let nodes = system
                .nodes
                .values()
                .map(|node| node.equiv)
                .choose_multiple(rng, size as _);
            assert!(nodes.len() == size as _);
            system.domains.push((nodes, self.rate_distr.sample(rng)));
            if (i + 1) % (self.num_domain / 100) == 0 {
                eprint!("\r{:<120}", format!("generated {} domains", i + 1))
            }
        }
    }
}

struct Merge<F, G>(F, G);

impl<F: FailureGenerator, G: FailureGenerator> FailureGenerator for Merge<F, G> {
    fn rate(&self, system: &mut System, rng: &mut impl Rng) {
        self.0.rate(system, rng);
        self.1.rate(system, rng)
    }
}

fn main() {
    let parameters = SystemParameters {
        num_node: 100_000,
        num_data: 100,
    };
    let failure_generator = IndependentFailure { rate: 1e-5 };
    let failure_generator = Merge(
        failure_generator,
        CorrelatedFailure {
            num_domain: 10_000,
            // average size = 100
            size_distr: Geometric::new(1. / 998.).expect("valid parameter"),
            rate_distr: Exp::new(1e4).expect("valid parameter"),
        },
    );
    // let mut protocol = NopProtocol { n: 3, k: 1 };
    let mut protocol = RingSuccessorProtocol { n: 12, k: 4 };

    let mut rng = StdRng::seed_from_u64(117418);
    let mut system = System::new(parameters);
    system.init(&failure_generator, &mut protocol, &mut rng);
    while system.step_count < 100_000 && !system.finalized() {
        system.step(&mut protocol, &mut rng);
        if system.step_count % 1_000 == 0 {
            system.print()
        }
    }
    eprintln!()
}
