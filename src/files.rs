use std::{path::Path, thread, time::Duration};


const WAITING_TIMEOUT_SECONDS: u64 = 120; 
const WAIT_SECONDS: u64 = 5;

fn wait(path: &Path) {
    let mut wait = 0;
    while !path.exists() {
        println!("File not found. Waiting {} seconds...", WAIT_SECONDS);
        thread::sleep(Duration::from_secs(WAIT_SECONDS));
        
        wait += WAIT_SECONDS;
        if wait == WAITING_TIMEOUT_SECONDS {
            eprintln!("Timed out while waiting for data.");
            std::process::exit(1);
        }
    };
}


pub fn read_dataslice(path: &String) -> std::collections::HashMap<String, tch::Tensor> {
    println!("Reading file...");
    let path_to_slice = std::path::Path::new(&path);
    wait(path_to_slice);
    let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
    match std::fs::remove_file(path_to_slice) {
        Ok(_) => {
            println!("Slice has been loaded.");
            dataslice.into_iter().collect()
        },
        Err(e) => panic!("Error deleting file: {:?}", e),
    }
}