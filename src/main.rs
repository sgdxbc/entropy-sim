use rand::{rngs::StdRng, seq::IteratorRandom, Rng, SeedableRng};

type HashMap<K, V> = rustc_hash::FxHashMap<K, V>;
type HashSet<V> = rustc_hash::FxHashSet<V>;

trait FailureGenerator {
    fn rate(&self, system: &mut System, rng: &mut impl Rng);
}

trait Protocol {
    fn enter(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng);

    fn repair(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng) -> bool;
}

type Digest = [u8; 32];

struct Node {
    equiv: NodeEquiv,
    fragments: Vec<(Digest, FragmentIndex)>,
}

type NodeEquiv = u32;
type FragmentIndex = usize;

#[derive(Default)]
struct System {
    step_count: u32,
    domains: Vec<(Vec<NodeEquiv>, f64)>,
    nodes: HashMap<NodeId, Node>,
    node_equiv: Vec<NodeId>,
    data: HashMap<Digest, HashMap<FragmentIndex, NodeId>>,
}

type NodeId = [u8; 32];

struct SystemParameters {
    num_node: usize,
    num_data: usize,
}

impl System {
    fn new() -> Self {
        Self::default()
    }

    fn init(
        &mut self,
        config: SystemParameters,
        failure_generator: &impl FailureGenerator,
        protocol: &mut impl Protocol,
        rng: &mut impl Rng,
    ) {
        self.node_equiv
            .resize_with(config.num_node, Default::default);
        for equiv in 0..config.num_node {
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
        for _ in 0..config.num_data {
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
        let mut involved_data = HashSet::default();
        for (nodes, failure_rate) in &self.domains {
            if !rng.gen_bool(*failure_rate) {
                continue;
            }
            for equiv in nodes {
                let id = &self.node_equiv[*equiv as usize];
                let Some(mut node) = self.nodes.remove(id) else {
                    continue;
                };
                for (digest, index) in node.fragments.drain(..) {
                    self.data
                        .get_mut(&digest)
                        .expect("data is not lost (yet)")
                        .remove(&index)
                        .expect("fragment is present");
                    involved_data.insert(digest);
                }
                let id = rng.gen();
                let replaced = self.nodes.insert(id, node);
                assert!(replaced.is_none());
                self.node_equiv[*equiv as usize] = id
            }
        }
        for digest in involved_data {
            if protocol.repair(&digest, self, rng) {
                continue;
            }
            println!("[step {:8}] data {digest:02x?} is lost", self.step_count);
            self.data
                .remove(&digest)
                .expect("data is not lost before failed repair");
        }
        self.step_count += 1
    }

    fn store(&mut self, digest: &Digest, index: FragmentIndex, id: &NodeId) {
        self.nodes
            .get_mut(id)
            .expect("node exists")
            .fragments
            .push((*digest, index)); // check multiple fragments on same node?
        let replaced = self
            .data
            .get_mut(digest)
            .expect("data exists")
            .insert(index, *id);
        assert!(replaced.is_none())
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
        for id in ids {
            system.store(data_digest, 0, &id)
        }
    }

    fn repair(&mut self, digest: &Digest, system: &mut System, _: &mut impl Rng) -> bool {
        system.data[digest].len() >= self.k
    }
}

struct IndependentFailure {
    rate: f64,
}

impl FailureGenerator for IndependentFailure {
    fn rate(&self, system: &mut System, _: &mut impl Rng) {
        for equiv in 0..system.nodes.len() {
            system.domains.push((vec![equiv as _], self.rate))
        }
    }
}

fn main() {
    let parameters = SystemParameters {
        num_node: 100_000,
        num_data: 100,
    };
    let failure_generator = IndependentFailure { rate: 1e-3 };
    let mut protocol = NopProtocol { n: 1, k: 1 };

    let mut rng = StdRng::seed_from_u64(117418);
    let mut system = System::new();
    system.init(parameters, &failure_generator, &mut protocol, &mut rng);
    while system.step_count < 1_000_000 && !system.finalized() {
        system.step(&mut protocol, &mut rng);
        if system.step_count % 10_000 == 0 {
            eprint!("\rstep {}", system.step_count)
        }
    }
}
