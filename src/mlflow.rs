

use reqwest;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

fn get_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}


#[derive(Serialize, Deserialize, Debug)]
pub struct Metric {
    key: String,
    value: f64,
    timestamp: u128,
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
            timestamp: get_epoch_ms(),
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
    pub user: String,
    pub password: String,
}

impl MLFlowClient {
    pub fn log_metrics(self, metrics: Vec<Metric>) {
        let body = RequestBody {
            run_id: self.run_id,
            metrics: metrics,
        };
        let client = reqwest::blocking::Client::new();
        match client.post(self.url)
                    .json(&body)
                    .basic_auth(self.user, Some(self.password))
                    .send() 
        {

             Ok(resp) => {
                // let err = resp.error_for_status();
                let status = resp.status();

                if status.is_client_error() {
                    let response_txt = resp.text().unwrap();
                    panic!("{}: {}", status.to_string(), response_txt);
                }
                
            }
             Err(err) => panic!("Error: {}", err)
         };
        
    }
}
