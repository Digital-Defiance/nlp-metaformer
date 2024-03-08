

use reqwest;
use serde::{Deserialize, Serialize};
use std::str::FromStr;


#[derive(Serialize, Deserialize, Debug)]
pub struct Metric {
    key: String,
    value: f64,
    step: i64
}

pub struct MetricAccumulator {
    key: String,
    value: f64,
    counter: i64
}

impl MetricAccumulator {
    pub fn new(key: &str) -> Self {
        Self { key: String::from_str(key).unwrap(), value: 0., counter: 0 }
    }

    pub fn accumulate(&mut self, value: f64) {
        self.value += value;
        self.counter += 1;
    }

    pub fn to_metric(self, step: i64) -> Metric {
        Metric {
            key: self.key,
            value: self.value / (self.counter as f64),
            step,
        }
    }   

}


#[derive(Serialize, Deserialize, Debug)]
struct RequestBody {
    run_id: String,
    metrics: Vec<Metric>,
}


pub struct MLFlowClient {
    pub run_id: String,
    pub url: String,
}

impl MLFlowClient {
    pub fn log_metrics(self, metrics: Vec<Metric>) {
        let body = RequestBody {
            run_id: self.run_id,
            metrics: metrics,
        };
        let client = reqwest::blocking::Client::new();
        match client.post(self.url).json(&body).send() {
             Ok(resp) => resp.text().unwrap(),
             Err(err) => panic!("Error: {}", err)
         };
        println!("Successfully logged metrics.");
    }
}
