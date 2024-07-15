use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    fmt::Write as _,
    fs::{create_dir_all, write},
    path::Path,
    time::{Duration, Instant, UNIX_EPOCH},
};

use rand::{rngs::StdRng, seq::IteratorRandom, Rng, SeedableRng};
use rand_distr::{Binomial, Distribution, Exp};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type HashMap<K, V> = rustc_hash::FxHashMap<K, V>;

trait FailureGenerator {
    fn rate(&self, system: &mut System, rng: &mut impl Rng);
}

trait Protocol {
    fn enter(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng);

    fn maintain(&mut self, data_digest: &Digest, system: &mut System, rng: &mut impl Rng) -> bool;

    fn name(&self) -> &'static str;

    fn redundancy(&self) -> f32;

    fn group_size(&self) -> usize;
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
    scheduled_failures: BinaryHeap<Reverse<(u32, usize)>>,

    config: SystemParameters,
}

type NodeId = [u8; 32];

#[derive(Clone)]
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
            scheduled_failures: Default::default(),
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
        failure_generator.rate(self, rng);
        for (domain_index, (_, failure_rate)) in self.domains.clone().iter().enumerate() {
            self.schedule(*failure_rate, rng, domain_index)
        }
    }

    fn finalized(&self) -> bool {
        self.data.is_empty()
    }

    fn step(
        &mut self,
        protocol: &mut impl Protocol,
        reporter: &mut impl Report,
        rng: &mut impl Rng,
    ) {
        let at = self
            .scheduled_failures
            .peek()
            .expect("exists scheduled failures")
            .0
             .0;
        assert!(at >= self.step_count, "{at} < {}", self.step_count);
        self.step_count = at;
        let Reverse((_, domain_index)) = self.scheduled_failures.pop().unwrap();
        let (nodes, failure_rate) = &self.domains[domain_index];
        for equiv in nodes {
            let id = &self.node_equiv[*equiv as usize];
            let Some(mut node) = self.nodes.remove(id) else {
                continue;
            };
            self.failure_count += 1;
            for (digest, index) in node.fragments.drain() {
                if let Some(fragments) = self.data.get_mut(&digest) {
                    fragments.remove(&index).expect("fragment is present");
                }
                // .expect("data is not lost (yet)")
                // .remove(&index)
            }
            let id = rng.gen();
            let replaced = self.nodes.insert(id, node);
            assert!(replaced.is_none());
            self.node_equiv[*equiv as usize] = id
        }
        self.schedule(*failure_rate, rng, domain_index);
        for digest in self.data.keys().cloned().collect::<Vec<_>>() {
            if protocol.maintain(&digest, self, rng) {
                continue;
            }
            self.data
                .remove(&digest)
                .expect("data is not lost before failed repair");
            reporter.report(&self.status())
        }
        // self.step_count += 1
    }

    fn schedule(&mut self, failure_rate: f64, rng: &mut impl Rng, domain_index: usize) {
        let interval = Exp::new(failure_rate)
            .expect("valid parameter")
            .sample(rng)
            .ceil() as u32;
        // assert!(self.step_count + interval >= self.step_count);
        if u32::MAX - interval > self.step_count {
            self.scheduled_failures
                .push(Reverse((self.step_count + interval, domain_index)))
        }
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

    fn status(&self) -> SystemStatus {
        SystemStatus {
            step: self.step_count,
            num_alive: self.data.len(),
            num_step_failure: self.failure_count as f32 / self.step_count as f32,
            churn: self.failure_count as f32 / self.config.num_node as f32 / self.step_count as f32
                * 100.,
            redundancy: self
                .data
                .values()
                .map(|fragments| fragments.len())
                .sum::<usize>() as f32
                / self.data.len() as f32,
        }
    }
}

struct SystemStatus {
    step: u32,
    num_alive: usize,
    num_step_failure: f32,
    churn: f32,
    redundancy: f32,
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

    fn name(&self) -> &'static str {
        "w/o repair"
    }

    fn redundancy(&self) -> f32 {
        self.n as f32 / self.k as f32
    }

    fn group_size(&self) -> usize {
        self.n
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

    fn name(&self) -> &'static str {
        "w/ repair"
    }

    fn redundancy(&self) -> f32 {
        self.n as f32 / self.k as f32
    }

    fn group_size(&self) -> usize {
        self.n
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
        eprint!("\rgenerated independent failures")
    }
}

struct CorrelatedFailure {
    num_domain: usize,
    size_distr: Binomial,
    rate_distr: Exp<f64>,
}

impl CorrelatedFailure {
    fn new(num_domain: usize, num_node: usize, mean_size: usize, failure_rate: f64) -> Self {
        assert!(mean_size > 2);
        Self {
            num_domain,
            size_distr: Binomial::new(num_node as _, mean_size as f64 / num_node as f64)
                .expect("valid parameter"),
            rate_distr: Exp::new(1. / failure_rate).expect("valid parameter"),
        }
    }
}

impl FailureGenerator for CorrelatedFailure {
    fn rate(&self, system: &mut System, rng: &mut impl Rng) {
        let mut period = Period(Instant::now());
        let mut sizes = Vec::new();
        for i in 0..self.num_domain {
            let mut size;
            while {
                size = self.size_distr.sample(rng);
                size < 2
            } {}
            sizes.push(size);
            let nodes = system
                .nodes
                .values()
                .map(|node| node.equiv)
                .choose_multiple(rng, size as _);
            assert!(nodes.len() == size as _);
            system.domains.push((nodes, self.rate_distr.sample(rng)));
            period.run(|| eprint!("\r{:<120}", format!("generated {} domains", i + 1)))
        }

        let average_size = sizes
            .into_iter()
            .map(|s| s as f64 / self.num_domain as f64)
            .sum::<f64>();
        eprintln!("\raverage domain size {average_size:.2}")
    }
}

struct Merge<F, G>(F, G);

impl<F: FailureGenerator, G: FailureGenerator> FailureGenerator for Merge<F, G> {
    fn rate(&self, system: &mut System, rng: &mut impl Rng) {
        self.0.rate(system, rng);
        self.1.rate(system, rng)
    }
}

struct Period(Instant);

impl Period {
    fn run(&mut self, task: impl FnOnce()) {
        if self.0.elapsed() > Duration::from_secs(1) {
            task();
            self.0 = Instant::now()
        }
    }
}

trait Report {
    fn report(&mut self, status: &SystemStatus);
}

struct OverridingLine;

impl Report for OverridingLine {
    fn report(&mut self, status: &SystemStatus) {
        eprint!(
            "{:<120}",
            format!(
                "\r[step {:7}] {:3} alive, {:4.2} failures/step ({:4.2}% churn), {:4.2} fragments/data",
                status.step,
                status.num_alive,
                status.num_step_failure,
                status.churn,
                status.redundancy,
            )
        )
    }
}

struct Csv {
    parameters: SystemParameters,
    domain_mean_size: usize,
    protocol: &'static str,
    redundancy: f32,
    group_size: usize,

    buf: String,
}

impl Report for Csv {
    fn report(&mut self, status: &SystemStatus) {
        writeln!(
            &mut self.buf,
            "{},{},{},{},{},{},{},{},{},{}",
            self.parameters.num_node,
            self.parameters.num_data,
            self.domain_mean_size,
            self.protocol,
            self.redundancy,
            self.group_size,
            status.step,
            status.num_alive,
            status.num_step_failure,
            status.churn,
        )
        .expect("writeln to string infallible")
    }
}

impl<F: Report, G: Report> Report for Merge<F, G> {
    fn report(&mut self, status: &SystemStatus) {
        self.0.report(status);
        self.1.report(status)
    }
}

fn main() {
    let parameters = SystemParameters {
        num_node: 1_000,
        num_data: 100,
    };
    let failure_generator = IndependentFailure { rate: 1e-5 };
    let mean_size = 120;
    let failure_generator = Merge(failure_generator, {
        let num_domain = 1_000_000;
        let num_step_failure = 20.;
        // num_domain * failure_rate * mean_size = num_step_failure
        CorrelatedFailure::new(
            num_domain,
            parameters.num_node,
            mean_size,
            num_step_failure / num_domain as f64 / mean_size as f64,
        )
    });
    // let mut protocol = NopProtocol { n: 12, k: 4 };
    let mut protocol = RingSuccessorProtocol { n: 12, k: 4 };

    // let mut reporter = OverridingLine;
    let csv = Csv {
        parameters: parameters.clone(),
        domain_mean_size: mean_size,
        protocol: protocol.name(),
        redundancy: protocol.redundancy(),
        group_size: protocol.group_size(),
        buf: Default::default(),
    };
    let mut reporter = Merge(OverridingLine, csv);

    let mut rng = StdRng::seed_from_u64(117418);
    let mut system = System::new(parameters);
    system.init(&failure_generator, &mut protocol, &mut rng);
    reporter.report(&system.status());

    let mut period = Period(Instant::now());
    while system.step_count < 1_000_000 && !system.finalized() {
        system.step(&mut protocol, &mut reporter, &mut rng);
        period.run(|| reporter.report(&system.status()))
    }
    reporter.report(&system.status());
    eprintln!();

    let data_dir = Path::new("data");
    create_dir_all(data_dir).expect("create directory is successful");
    write(
        data_dir.join(format!(
            "{}.csv",
            UNIX_EPOCH
                .elapsed()
                .expect("UNIX epoch is elapsed")
                .as_secs()
        )),
        reporter.1.buf,
    )
    .expect("write data is successful")
}
