//! Limit Order Book reconstruction from a stream of order events.

pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
}

pub struct OrderBookSnapshot {
    pub timestamp_ms: u64,
    pub bids: Vec<PriceLevel>,  // sorted descending by price
    pub asks: Vec<PriceLevel>,  // sorted ascending by price
}

impl OrderBookSnapshot {
    pub fn mid_price(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            None
        } else {
            Some((self.bids[0].price + self.asks[0].price) / 2.0)
        }
    }

    pub fn spread(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            None
        } else {
            Some(self.asks[0].price - self.bids[0].price)
        }
    }
}

pub struct OrderBookReconstructor {
    bids: std::collections::BTreeMap<std::cmp::Reverse<ordered_float::OrderedFloat<f64>>, f64>,
    asks: std::collections::BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
}

impl OrderBookReconstructor {
    pub fn new() -> Self {
        Self {
            bids: std::collections::BTreeMap::new(),
            asks: std::collections::BTreeMap::new(),
        }
    }

    pub fn update_level(&mut self, side: &str, price: f64, size: f64) {
        match side {
            "bid" => {
                if size == 0.0 {
                    self.bids.remove(&std::cmp::Reverse(ordered_float::OrderedFloat(price)));
                } else {
                    self.bids.insert(std::cmp::Reverse(ordered_float::OrderedFloat(price)), size);
                }
            }
            "ask" => {
                if size == 0.0 {
                    self.asks.remove(&ordered_float::OrderedFloat(price));
                } else {
                    self.asks.insert(ordered_float::OrderedFloat(price), size);
                }
            }
            _ => {}
        }
    }

    pub fn snapshot(&self, timestamp_ms: u64, levels: usize) -> OrderBookSnapshot {
        let bids: Vec<PriceLevel> = self.bids.iter()
            .take(levels)
            .map(|(p, s)| PriceLevel { price: p.0.into(), size: *s })
            .collect();

        let asks: Vec<PriceLevel> = self.asks.iter()
            .take(levels)
            .map(|(p, s)| PriceLevel { price: p.into_inner(), size: *s })
            .collect();

        OrderBookSnapshot { timestamp_ms, bids, asks }
    }
}

// Simple ordered float wrapper (avoids extra dependency in MVP)
mod ordered_float {
    use std::cmp::Ordering;

    #[derive(Clone, Copy, Debug)]
    pub struct OrderedFloat(pub f64);

    impl PartialEq for OrderedFloat {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl Eq for OrderedFloat {}

    impl PartialOrd for OrderedFloat {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl Ord for OrderedFloat {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
        }
    }

    impl From<OrderedFloat> for f64 {
        fn from(of: OrderedFloat) -> f64 {
            of.0
        }
    }

    impl OrderedFloat {
        pub fn into_inner(self) -> f64 {
            self.0
        }
    }
}
