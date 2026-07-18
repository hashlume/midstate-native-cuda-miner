use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Job {
    pub generation: u64,
    pub id: u64,
    pub midstate: [u8; 32],
    pub network_target: [u8; 32],
}

pub fn authorize(id: u64, address: &str, worker: &str) -> Value {
    json!({
        "id": id,
        "method": "mining.authorize",
        "params": [address, worker],
    })
}

pub fn submit(id: u64, address: &str, job_id: u64, nonce: u64, final_hash: [u8; 32]) -> Value {
    json!({
        "id": id,
        "method": "mining.submit",
        "params": [address, job_id, nonce, hex::encode(final_hash)],
    })
}

pub fn parse_job(message: &Value, generation: u64) -> Result<Option<Job>> {
    if message.get("method").and_then(Value::as_str) != Some("mining.notify") {
        return Ok(None);
    }
    let params = message
        .get("params")
        .and_then(Value::as_array)
        .context("mining.notify params must be an array")?;
    if params.len() < 3 {
        bail!("mining.notify requires job id, midstate, and batch template");
    }

    let id = params[0].as_u64().context("job id must be u64")?;
    let midstate_hex = params[1].as_str().context("midstate must be hex")?;
    let mut midstate = [0u8; 32];
    hex::decode_to_slice(midstate_hex, &mut midstate).context("invalid 32-byte midstate")?;

    let target_values = params[2]
        .get("target")
        .and_then(Value::as_array)
        .context("batch template is missing target")?;
    if target_values.len() != 32 {
        bail!("network target must contain 32 bytes");
    }
    let mut network_target = [0u8; 32];
    for (index, value) in target_values.iter().enumerate() {
        let byte = value
            .as_u64()
            .ok_or_else(|| anyhow!("network target byte {index} is not an integer"))?;
        network_target[index] =
            u8::try_from(byte).map_err(|_| anyhow!("network target byte {index} exceeds 255"))?;
    }

    Ok(Some(Job {
        generation,
        id,
        midstate,
        network_target,
    }))
}

pub fn response_id(message: &Value) -> Option<u64> {
    message.get("id").and_then(Value::as_u64)
}

pub fn response_accepted(message: &Value) -> Option<bool> {
    message.get("result").and_then(Value::as_bool)
}

pub fn response_error(message: &Value) -> Option<String> {
    match message.get("error") {
        None | Some(Value::Null) => None,
        Some(Value::String(error)) => Some(error.clone()),
        Some(error) => Some(error.to_string()),
    }
}

pub fn below_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    hash < target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_live_format_job() {
        let target: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let message = json!({
            "id": null,
            "method": "mining.notify",
            "params": [42, "aa".repeat(32), { "target": target }],
        });
        let job = parse_job(&message, 7).unwrap().unwrap();
        assert_eq!(job.generation, 7);
        assert_eq!(job.id, 42);
        assert_eq!(job.midstate, [0xaa; 32]);
        assert_eq!(job.network_target[31], 31);
    }

    #[test]
    fn orders_hashes_as_big_endian_targets() {
        let mut target = [0xff; 32];
        target[0] = 1;
        let mut valid = [0xff; 32];
        valid[0] = 0;
        assert!(below_target(&valid, &target));
        assert!(!below_target(&target, &target));
    }
}
