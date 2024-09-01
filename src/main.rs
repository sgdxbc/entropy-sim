use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
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
    // let r = 2.;
    // let r = 2.4;
    let parameters = SystemParameters {
        num_initial_node: num_node,
        // enter_rate: 0.01, // node/step, not % node
        enter_rate: 0.02,
        leave_rate: 0.01,
        target_redundancy: r,
        num_chunk: k,
        // num_census_step: None,
        // num_census_step: Some(1_000_000),
        num_census_step: Some(10_000),
        num_checkpoint_step: 100_000,
        // num_data: 400,
        // num_data: 200,
        num_data: 100,
    };

    let mut system = System::new(parameters);
    if args().nth(1).as_deref() == Some("dump-degree") {
        dump_degree(&system);
        return;
    }
    let mut rng = StdRng::seed_from_u64(117418);
    system.init(&mut rng);
    eprintln!("{}", system.report());
    let mut period = Period(Instant::now());
    // while system.now < 1_000_000_000 {
    while system.now < 10_000_000 {
        system.step(&mut rng);
        period.run(|| eprint!("{:120}\r", system.report()))
    }
    eprint!("{:120}\r", system.report());
    eprintln!();
    if args().nth(1).as_deref() == Some("dump-load") {
        dump_load(&system)
    }
    if args().nth(1).as_deref() == Some("dump-checkpoint") {
        system.parameters.enter_rate = system.parameters.leave_rate;
        while system.now < 12_000_000 {
            system.step(&mut rng);
            period.run(|| eprint!("{:120}\r", system.report()))
        }
        eprint!("{:120}\r", system.report());
        eprintln!();

        dump_checkpoint(&system)
    }
}

fn dump_degree(system: &System) {
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
}

fn dump_load(system: &System) {
    let mut lines = String::new();
    for (i, load) in system.storage_loads.iter().enumerate() {
        writeln!(
            &mut lines,
            "{},{i},{load:.2}",
            system.parameters.comma_separated()
        )
        .expect("can write")
    }
    let name = format!("load-{}", Local::now().format("%y%m%d%H%M%S"));
    let path = Path::new("data").join(name).with_extension("csv");
    write(path, lines).expect("write results to file");
}

fn dump_checkpoint(system: &System) {
    let mut lines = String::new();
    for checkpoint in &system.checkpoints {
        writeln!(
            &mut lines,
            "{},{},{},{:.2}",
            system.parameters.comma_separated(),
            checkpoint.at,
            checkpoint.num_node,
            checkpoint.redundancy,
        )
        .expect("can write")
    }
    let name = format!("checkpoint-{}", Local::now().format("%y%m%d%H%M%S"));
    let path = Path::new("data").join(name).with_extension("csv");
    write(path, lines).expect("write results to file");
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
    num_census_step: Option<u32>,
    // evaluation
    num_checkpoint_step: u32,
    num_data: usize,
}

impl SystemParameters {
    fn comma_separated(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{}",
            self.num_initial_node,
            self.enter_rate,
            self.leave_rate,
            self.target_redundancy,
            self.num_chunk,
            self.num_census_step
                .map(|n| n.to_string())
                .unwrap_or("null".into()),
            self.num_checkpoint_step,
            self.num_data,
        )
    }
}

type Step = u32;
type DataId = u32;
#[allow(unused)]
type ChunkId = u32;
// type Packet = HashSet<ChunkId>; // need back reference to DataId?
type Packet = usize; // the degree of packet
type NodeId = u32;

#[derive(Debug)]
struct System {
    parameters: SystemParameters,
    packet_distr: PacketDistr,

    now: Step,
    events: BinaryHeap<Reverse<(Step, Event)>>,
    nodes: Vec<Node>,
    node_id_count: NodeId,

    num_node_enter: u32,
    num_node_leave: u32,
    storage_loads: Vec<f32>,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Default)]
struct Node {
    id: NodeId,
    packets: HashMap<DataId, Vec<Packet>>,
}

#[derive(Debug)]
struct Checkpoint {
    at: Step,
    num_node: usize,
    redundancy: f32,
}

impl Node {
    // in the unit of chunk size i.e. (1 / num_chunk) unit size
    // 1 unit size = the size of 1 data
    fn storage_size(&self) -> usize {
        self.packets
            .values()
            .map(|packets| packets.iter().sum::<usize>())
            .sum()
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    NodeEnter,
    NodeLeave,
    AdjustRedundancy(usize, NodeId),
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
            node_id_count: 0,
            num_node_enter: 0,
            num_node_leave: 0,
            storage_loads: Default::default(),
            checkpoints: Default::default(),
        }
    }

    fn init(&mut self, rng: &mut impl Rng) {
        // a bit unrealistic here: initialize the system by assuming perfect scale estimation by
        // every initial node
        let num_node_packet = self.parameters.num_chunk as f64 * self.parameters.target_redundancy
            / self.parameters.num_initial_node as f64;
        for _ in 0..self.parameters.num_initial_node {
            self.push_node(rng, num_node_packet)
        }
        self.push_checkpoint();
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

                // TODO introduce estimation error
                let num_node_packet = self.parameters.num_chunk as f64
                    * self.parameters.target_redundancy
                    / self.nodes.len() as f64;
                self.push_node(rng, num_node_packet);
                self.push_checkpoint();
                self.push_node_enter_event(rng)
            }
            NodeLeave => {
                self.num_node_leave += 1;
                //
                self.nodes.swap_remove(rng.gen_range(0..self.nodes.len()));
                self.push_checkpoint();
                self.push_node_leave_event(rng)
            }
            AdjustRedundancy(node_index, node_id) => {
                // TODO introduce estimation error
                let num_node_packet = (self.parameters.num_chunk as f64
                    * self.parameters.target_redundancy
                    / self.nodes.len() as f64)
                    .ceil() as usize;
                if let Some(node) = self.nodes.get_mut(node_index) {
                    if node.id == node_id {
                        for packets in node.packets.values_mut() {
                            // TODO increase redundancy
                            while packets.len() > num_node_packet {
                                packets.swap_remove(rng.gen_range(0..packets.len()));
                            }
                        }
                        self.push_adjust_redundancy_event(node_index, false, rng)
                    }
                }
            }
        }
    }

    fn push_node(&mut self, rng: &mut impl Rng, num_node_packet: f64) {
        let mut packets = HashMap::new();
        for data_id in 0..self.parameters.num_data {
            let mut data_packets = repeat_with(|| self.packet_distr.sample(rng))
                .take(num_node_packet.floor() as _)
                .collect::<Vec<_>>();
            if rng.gen_bool(num_node_packet.fract()) {
                data_packets.push(self.packet_distr.sample(rng))
            }
            packets.insert(data_id as DataId, data_packets);
        }
        self.node_id_count += 1;
        let id = self.node_id_count;
        let node = Node { id, packets };
        self.storage_loads
            .push(node.storage_size() as f32 / self.parameters.num_chunk as f32);
        let node_index = self.nodes.len();
        self.nodes.push(node);
        // not the best way to check for `delay`, but should be good enough
        self.push_adjust_redundancy_event(node_index, self.now == 0, rng)
    }

    fn push_checkpoint(&mut self) {
        if let Some(checkpoint) = self.checkpoints.last() {
            if checkpoint.at + self.parameters.num_checkpoint_step >= self.now {
                return;
            }
        }
        // eprintln!("checkpoint at {}", self.now);
        let checkpoint = Checkpoint {
            at: self.now,
            num_node: self.nodes.len(),
            redundancy: self.storage_size() as f32
                / self.parameters.num_chunk as f32
                / self.parameters.num_data as f32,
        };
        self.checkpoints.push(checkpoint)
    }

    fn report(&self) -> String {
        let num_packets = self
            .nodes
            .iter()
            .map(|node| {
                node.packets
                    .values()
                    .map(|packets| packets.len())
                    .sum::<usize>()
            })
            .sum::<usize>();
        let num_data_packet = num_packets as f32 / self.parameters.num_data as f32;
        let storage_size = self.storage_size();
        format!(
            concat!(
                "[{:>10}] {} node(s) {:.2}%/{:.2}% enter/leave (per kStep) {:.2}/{:.2} logical/real redundant {:.2} degree",
                " {:.2} average load"),
            self.now,
            self.nodes.len(),
            self.num_node_enter as f32 / (self.now as f32 / 1000.) / self.nodes.len() as f32 * 100.,
            self.num_node_leave as f32 / (self.now as f32 / 1000.) / self.nodes.len() as f32 * 100.,
            num_data_packet / self.parameters.num_chunk as f32,
            storage_size as f32
                / self.parameters.num_chunk as f32
                / self.parameters.num_data as f32,
            storage_size as f32 / num_packets as f32,
            storage_size as f32 / self.parameters.num_chunk as f32 / self.nodes.len() as f32,
        )
    }

    fn storage_size(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.storage_size())
            .sum::<usize>()
    }

    fn push_event(&mut self, after: Step, event: Event) {
        let Some(at) = self.now.checked_add(after) else {
            eprintln!("Timestamp overflow {event:?}");
            return;
        };
        self.events.push(Reverse((at, event)))
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

    fn push_adjust_redundancy_event(&mut self, node_index: usize, delay: bool, rng: &mut impl Rng) {
        let Some(mut after) = self.parameters.num_census_step else {
            return;
        };
        if delay {
            after += rng.gen_range(0..after)
        }
        self.push_event(
            after,
            Event::AdjustRedundancy(node_index, self.nodes[node_index].id),
        )
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
struct PacketDistr(WeightedIndex<f64>, #[allow(unused)] Uniform<u32>);

impl PacketDistr {
    fn new(k: usize) -> Self {
        Self(
            WeightedIndex::new(p_degree(k)).expect("valid weights"),
            Uniform::new(0, k as u32).expect("valid range"),
        )
    }

    // fn sample(&self, rng: &mut impl Rng) -> Packet {
    //     let d = self.0.sample(rng) + 1;
    //     let mut fragment = HashSet::new();
    //     while fragment.len() < d {
    //         fragment.insert(self.1.sample(rng));
    //     }
    //     fragment
    // }

    fn sample(&self, rng: &mut impl Rng) -> Packet {
        self.0.sample(rng) + 1
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
