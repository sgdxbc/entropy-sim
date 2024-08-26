use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Uniform, WeightedIndex};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type DegreeDistr = WeightedIndex<f64>;

fn main() {
    let num_node = 100_000;
    let r = 1.6;

    let k = (num_node as f32 / r) as usize;
    println!("k = {k}");
    let degree_weights = p_degree(k);
    let degree_distr = WeightedIndex::new(degree_weights).expect("valid weights");

    let mut rng = StdRng::seed_from_u64(117418);
    let mut decoder = Decoder::new(k);
    let mut period = Period(Instant::now());
    for i in 0.. {
        let fragment = sample_fragment(k, &degree_distr, &mut rng);
        // println!("sampled = {fragment:?}");
        decoder.receive(fragment);
        if decoder.recovered() {
            println!("recovered with {} fragment(s)", i + 1);
            break;
        }
        period.run(|| println!("received {} / {}", i + 1, decoder.received.len()))
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

fn sample_fragment(k: usize, degree_distr: &DegreeDistr, rng: &mut impl Rng) -> Vec<usize> {
    let d = sample_degree(degree_distr, rng);
    Uniform::new(0, k).sample_iter(rng).take(d).collect()
}

struct Decoder {
    buf: Vec<HashSet<usize>>,
    received: HashSet<usize>,
    k: usize,
}

impl Decoder {
    fn new(k: usize) -> Self {
        Self {
            k,
            buf: Default::default(),
            received: Default::default(),
        }
    }

    fn recovered(&self) -> bool {
        self.received.len() == self.k
    }

    fn receive(&mut self, fragment: Vec<usize>) {
        let fragment = fragment
            .into_iter()
            .filter(|i| !self.received.contains(i))
            .collect::<HashSet<_>>();
        if fragment.is_empty() {
            return;
        }
        if fragment.len() > 1 {
            self.buf.push(fragment) // dedup?
        } else {
            let mut new_received = vec![fragment.into_iter().next().unwrap()];
            while let Some(index) = new_received.pop() {
                self.received.insert(index);
                let mut i = 0;
                while i < self.buf.len() {
                    self.buf[i].remove(&index);
                    if self.buf[i].len() == 1 {
                        let fragment = self.buf.swap_remove(i);
                        new_received.push(fragment.into_iter().next().unwrap())
                    } else {
                        i += 1
                    }
                }
            }
        }
    }
}
