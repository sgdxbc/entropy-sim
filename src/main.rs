use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    env::args,
    fmt::Write,
    fs::write,
    iter::repeat_with,
    path::Path,
    time::{Duration, Instant},
};

use chrono::Local;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Uniform, WeightedIndex};

#[allow(unused)]
mod decoder;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let num_node = 100_000;
    // 640K chunks per data, 64GB data = 100KB chunk, 1GB data = 1.6KB chunk
    // the robust soliton distribution "bursts" at 5688 i.e. 568.8MB storage overhead for 64GB data
    let k = 64 * 10_000;
    // let k = 32 * 10_000;
    // let k = 16 * 10_000;
    let r = 1.6;
    let parameters = SystemParameters {
        num_initial_node: num_node,
        enter_rate: 1., // node/step, not % node
        leave_rate: 1.,
        target_redundancy: r,
        num_chunk: k,
    };

    let mut system = System::new(parameters);
    if args().nth(1).as_deref() == Some("dump-degree") {
        let mut lines = String::new();
        system.packet_distr.write_degree_distr(&mut lines);
        let lines = lines
            .lines()
            .map(|line| system.parameters.comma_separated() + "," + line)
            .collect::<Vec<_>>()
            .join("\n");
        let name = format!("degree-{}", Local::now().format("%y%m%d%H%M%S"));
        let path = Path::new("data").join(name).with_extension("csv");
        write(path, lines).expect("write results to file");
        return;
    }
    let mut rng = StdRng::seed_from_u64(117418);
    system.init(&mut rng);
    let mut period = Period(Instant::now());
    while system.now < 1_000_000_000 {
        system.step(&mut rng);
        period.run(|| eprint!("{:80}\r", system.report()))
    }
    eprint!("{:80}\r", system.report());
    eprintln!()
}

#[derive(Debug)]
struct SystemParameters {
    // network
    num_initial_node: usize,
    enter_rate: f64,
    leave_rate: f64,
    // protocol
    target_redundancy: f64, // > 1, tolerate (1 - 1 / _) portion of faulty nodes
    num_chunk: usize,       // k
}

impl SystemParameters {
    fn comma_separated(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.num_initial_node,
            self.enter_rate,
            self.leave_rate,
            self.target_redundancy,
            self.num_chunk
        )
    }
}

type Step = u32;
type DataId = u32;
type ChunkId = u32;
type Packet = HashSet<ChunkId>; // need DataId as well?

#[derive(Debug)]
struct System {
    parameters: SystemParameters,
    packet_distr: PacketDistr,

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
        let packet_distr = PacketDistr::new(parameters.num_chunk);
        Self {
            parameters,
            packet_distr,
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
        assert!(at >= self.now);
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
            "[{:>8}] {} node(s) {:.2}% enter/kStep {:.2}% leave/kStep",
            self.now,
            self.nodes.len(),
            self.num_node_enter as f32 / (self.now as f32 / 1000.) * 100.,
            self.num_node_leave as f32 / (self.now as f32 / 1000.) * 100.,
        )
    }

    fn push_event(&mut self, after: Step, event: Event) {
        let Some(at) = self.now.checked_add(after) else {
            eprintln!("Timestamp overflow {event:?}");
            return;
        };
        self.events.push(Reverse((at, event)))
    }

    fn push_node_enter_event(&mut self, rng: &mut impl Rng) {
        let after = Exp::new(self.parameters.enter_rate / self.nodes.len() as f64)
            .expect("valid parameter")
            .sample(rng)
            .ceil() as _;
        self.push_event(after, Event::NodeEnter)
    }

    fn push_node_leave_event(&mut self, rng: &mut impl Rng) {
        let after = Exp::new(self.parameters.leave_rate / self.nodes.len() as f64)
            .expect("valid parameter")
            .sample(rng)
            .ceil() as _;
        self.push_event(after, Event::NodeLeave)
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
    eprintln!("k {k} s {s} k/s {}", (k / s).floor());
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

#[derive(Debug)]
struct PacketDistr(WeightedIndex<f64>, Uniform<u32>);

impl PacketDistr {
    fn new(k: usize) -> Self {
        Self(
            WeightedIndex::new(p_degree(k)).expect("valid weights"),
            Uniform::new(0, k as u32).expect("valid range"),
        )
    }

    fn sample(&self, rng: &mut impl Rng) -> Packet {
        let d = self.0.sample(rng) + 1;
        let mut fragment = HashSet::new();
        while fragment.len() < d {
            fragment.insert(self.1.sample(rng));
        }
        fragment
    }

    fn write_degree_distr(&self, mut write: impl Write) {
        for (i, weight) in self.0.weights().enumerate() {
            writeln!(&mut write, "{},{}", i + 1, weight / self.0.total_weight()).expect("can write")
        }
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
