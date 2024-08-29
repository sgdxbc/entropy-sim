use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    iter::repeat_with,
    time::{Duration, Instant},
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp, WeightedIndex};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type DegreeDistr = WeightedIndex<f64>;

fn main() {
    let num_node = 100_000;
    let num_node_packet = 10;
    let r = 1.6;
    let parameters = SystemParameters {
        num_initial_node: num_node,
        enter_rate: 0.1,
        leave_rate: 0.1,
    };

    let k = ((num_node * num_node_packet) as f32 / r) as usize;
    println!("k = {k}");
    // let degree_distr = WeightedIndex::new(p_degree(k)).expect("valid weights");
    let mut rng = StdRng::seed_from_u64(117418);
    // let mut decoder = Decoder::new(k);

    let mut system = System::new(parameters);
    system.init(&mut rng);
    let mut period = Period(Instant::now());
    while system.now < 10_000_000 {
        system.step(&mut rng);
        period.run(|| eprint!("{:80}\r", system.report()))
    }
    eprint!("{:80}\r", system.report());
    eprintln!()
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

fn p_degree_ideal(k: usize) -> Vec<f64> {
    let mut ps = vec![1. / k as f64];
    for i in 2..=k {
        ps.push(1. / (i * (i - 1)) as f64)
    }
    ps
}

fn p_degree(k: usize) -> Vec<f64> {
    let delta = 0.5;
    let c = 0.01;

    let mut tau = p_degree_ideal(k);
    let k = k as f64;
    let s = c * (k / delta).ln() * k.sqrt();
    for (i, p) in tau.iter_mut().enumerate() {
        let d = (i + 1) as f64;
        if d == (k / s).floor() {
            *p += s / k * (s / delta).ln();
            break;
        }
        *p += s / k * (1. / d)
    }
    tau
}

fn sample_degree(degree_distr: &DegreeDistr, rng: &mut impl Rng) -> usize {
    degree_distr.sample(rng) + 1
}

fn sample_fragment(k: usize, degree_distr: &DegreeDistr, rng: &mut impl Rng) -> HashSet<usize> {
    let d = sample_degree(degree_distr, rng);
    let mut fragment = HashSet::new();
    while fragment.len() < d {
        fragment.insert(rand_distr::Uniform::new(0, k).sample(rng));
    }
    fragment
}

struct Decoder {
    count: u32,
    buf: HashMap<u32, HashSet<usize>>,
    pending: HashMap<usize, HashSet<u32>>,
    received: HashSet<usize>,
    k: usize,
}

impl Decoder {
    fn new(k: usize) -> Self {
        Self {
            k,
            count: 0,
            buf: Default::default(),
            pending: Default::default(),
            received: Default::default(),
        }
    }

    fn recovered(&self) -> bool {
        self.received.len() == self.k
    }

    fn receive(&mut self, fragment: HashSet<usize>) {
        let fragment = &fragment - &self.received;
        if fragment.is_empty() {
            return;
        }
        if fragment.len() > 1 {
            self.count += 1;
            let id = self.count;
            for &index in &fragment {
                self.pending.entry(index).or_default().insert(id);
            }
            self.buf.insert(id, fragment); // dedup?
        } else {
            let mut new_received = vec![fragment.into_iter().next().unwrap()];
            while let Some(index) = new_received.pop() {
                self.received.insert(index);
                for id in self.pending.remove(&index).unwrap_or_default() {
                    if let Some(fragment) = self.buf.get_mut(&id) {
                        fragment.remove(&index);
                        if fragment.len() == 1 {
                            let fragment = self.buf.remove(&id).unwrap();
                            new_received.extend(fragment)
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct SystemParameters {
    num_initial_node: usize,
    enter_rate: f64,
    leave_rate: f64,
}

type Step = u32;
type DataId = u32;
type ChunkId = u32;
type Packet = HashSet<ChunkId>; // need DataId as well?

#[derive(Debug)]
struct System {
    parameters: SystemParameters,
    now: Step,
    events: BinaryHeap<Reverse<(Step, Event)>>,
    nodes: Vec<Node>,

    num_node_enter: u32,
    num_node_leave: u32,
}

#[derive(Debug, Default)]
struct Node {
    packets: HashMap<DataId, Vec<Packet>>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    NodeEnter,
    NodeLeave,
}

impl System {
    fn new(parameters: SystemParameters) -> Self {
        Self {
            parameters,
            now: 0,
            events: Default::default(),
            nodes: Default::default(),

            num_node_enter: 0,
            num_node_leave: 0,
        }
    }

    fn init(&mut self, rng: &mut impl Rng) {
        self.nodes
            .extend(repeat_with(Node::default).take(self.parameters.num_initial_node));
        self.push_node_enter_event(rng);
        self.push_node_leave_event(rng)
    }

    fn step(&mut self, rng: &mut impl Rng) {
        let Reverse((at, event)) = self.events.pop().expect("events not exhausted");
        self.now = at;
        use Event::*;
        match event {
            NodeEnter => {
                self.num_node_enter += 1;
                // TODO
                let node = Node {
                    packets: Default::default(),
                };
                self.nodes.push(node);
                self.push_node_enter_event(rng)
            }
            NodeLeave => {
                self.num_node_leave += 1;
                // TODO
                self.nodes.swap_remove(rng.gen_range(0..self.nodes.len()));
                self.push_node_leave_event(rng)
            }
        }
    }

    fn report(&self) -> String {
        format!(
            "[{:>8}] {} node(s) {:.2}% enter/step {:.2}% leave/step",
            self.now,
            self.nodes.len(),
            self.num_node_enter as f32 / self.now as f32 * 100.,
            self.num_node_leave as f32 / self.now as f32 * 100.,
        )
    }

    fn push_event(&mut self, after: Step, event: Event) {
        self.events.push(Reverse((self.now + after, event)))
    }

    fn push_node_enter_event(&mut self, rng: &mut impl Rng) {
        let after = Exp::new(self.parameters.enter_rate)
            .expect("valid parameter")
            .sample(rng)
            .ceil() as _;
        self.push_event(after, Event::NodeEnter)
    }

    fn push_node_leave_event(&mut self, rng: &mut impl Rng) {
        let after = Exp::new(self.parameters.leave_rate)
            .expect("valid parameter")
            .sample(rng)
            .ceil() as _;
        self.push_event(after, Event::NodeLeave)
    }
}
