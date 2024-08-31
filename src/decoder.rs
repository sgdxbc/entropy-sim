use std::collections::{HashMap, HashSet};

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
