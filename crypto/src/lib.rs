//! GARY Phase 5 — the recoverability boundary ("decypher total").
//!
//! This extends Phase 3 (lingua cosmica) into the cryptographic regime. Phase 3 showed an
//! emergent code is decipherable from an anchor (behaviour or cribs). Here we contrast:
//!   - a WEAK code (a fixed substitution): recoverable from cribs (known plaintext);
//!   - a KEYED code (a per-nonce stream shift): NOT recoverable from cribs, because the
//!     mapping changes every message and the keystream is unpredictable without the key.
//!
//! An eavesdropper with known-plaintext on some messages is evaluated on HELD-OUT messages
//! (unseen nonces). The substitution falls; the keyed code holds at chance. WITH the key the
//! keyed code decodes perfectly — its resistance is real secrecy, not garbled data.
//!
//! NOTE: the keyed stream below is a RESEARCH stand-in (a keyed counter PRNG), used only to
//! demonstrate the recoverability gap — it is NOT production cryptography. The real PQC pillar
//! (ML-DSA signing of result artifacts) will use a vetted crate.

use std::collections::HashMap;

use fips204::ml_dsa_65;
use fips204::traits::{Signer, Verifier};

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Tiny deterministic PRNG (splitmix64 counter) — keeps the crate dependency-free.
pub struct Prng {
    state: u64,
}
impl Prng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        splitmix64(self.state)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Per-nonce keystream. RESEARCH stand-in (keyed PRNG), NOT production crypto.
fn keystream(key: u64, nonce: u64) -> u64 {
    splitmix64(key ^ splitmix64(nonce))
}

/// Weak code: a fixed substitution permutation. The same state maps to the same symbol every
/// message, so cribs recover the whole table (monoalphabetic, historically breakable).
pub struct SubstitutionCode {
    perm: Vec<u8>,
}
impl SubstitutionCode {
    pub fn new(n: u8, seed: u64) -> Self {
        let mut perm: Vec<u8> = (0..n).collect();
        let mut rng = Prng::new(seed);
        for i in (1..n as usize).rev() {
            let j = rng.below((i + 1) as u64) as usize;
            perm.swap(i, j);
        }
        Self { perm }
    }
    pub fn encode(&self, state: u8, _nonce: u64) -> u8 {
        self.perm[state as usize]
    }
}

/// Keyed code: state shifted by a per-nonce keystream value over Z_n (one-time-pad-like).
/// Without the key the per-nonce shift is unpredictable, so cribs do not generalise.
pub struct KeyedStreamCode {
    key: u64,
    n: u8,
}
impl KeyedStreamCode {
    pub fn new(key: u64, n: u8) -> Self {
        Self { key, n }
    }
    pub fn encode(&self, state: u8, nonce: u64) -> u8 {
        ((state as u64 + keystream(self.key, nonce)) % self.n as u64) as u8
    }
}

pub struct CryptResult {
    pub substitution_acc: f64,    // eavesdropper accuracy on held-out, weak code
    pub keyed_acc: f64,           // eavesdropper accuracy on held-out, keyed code (no key)
    pub keyed_with_key_acc: f64,  // accuracy when the eavesdropper has the key (firewall)
}

/// Run the cryptanalysis: an eavesdropper learns from `crib_fraction` known-plaintext on the
/// training messages, then is scored on held-out messages with UNSEEN nonces.
pub fn cryptanalysis(
    n: u8,
    train_msgs: usize,
    test_msgs: usize,
    crib_fraction: f64,
    seed: u64,
) -> CryptResult {
    let key = seed ^ 0xDEAD_BEEF_CAFE_F00D;
    let sub = SubstitutionCode::new(n, seed);
    let keyed = KeyedStreamCode::new(key, n);
    let mut rng = Prng::new(seed ^ 0x1234_5678);

    // What the eavesdropper learns from cribs.
    let mut sub_table: Vec<Option<u8>> = vec![None; n as usize]; // symbol -> state (stable)
    let mut keyed_shift: HashMap<u64, u8> = HashMap::new(); // nonce -> shift (does NOT generalise)

    for i in 0..train_msgs {
        let state = rng.below(n as u64) as u8;
        let nonce = i as u64;
        let sub_sym = sub.encode(state, nonce);
        let keyed_sym = keyed.encode(state, nonce);
        let is_crib = (rng.next_u64() as f64 / u64::MAX as f64) < crib_fraction;
        if is_crib {
            sub_table[sub_sym as usize] = Some(state);
            let shift = (keyed_sym as i64 - state as i64).rem_euclid(n as i64) as u8;
            keyed_shift.insert(nonce, shift);
        }
    }

    // Held-out test: unseen nonces, so the keyed eavesdropper has no shift for them.
    let mut sub_ok = 0usize;
    let mut keyed_ok = 0usize;
    let mut keyed_key_ok = 0usize;
    for j in 0..test_msgs {
        let state = rng.below(n as u64) as u8;
        let nonce = (train_msgs + j) as u64;
        let sub_sym = sub.encode(state, nonce);
        let keyed_sym = keyed.encode(state, nonce);

        // Weak: invert via the learned symbol->state table (fall back to a fixed guess).
        let sub_guess = sub_table[sub_sym as usize].unwrap_or(0);
        if sub_guess == state {
            sub_ok += 1;
        }

        // Keyed, no key: this nonce was never cribbed -> no shift -> a fixed guess (chance).
        let keyed_guess = match keyed_shift.get(&nonce) {
            Some(&shift) => (keyed_sym as i64 - shift as i64).rem_euclid(n as i64) as u8,
            None => 0u8,
        };
        if keyed_guess == state {
            keyed_ok += 1;
        }

        // Keyed, WITH key: recompute the shift -> perfect decode (firewall: real secrecy).
        let true_shift = (keystream(key, nonce) % n as u64) as u8;
        let keyed_key_guess = (keyed_sym as i64 - true_shift as i64).rem_euclid(n as i64) as u8;
        if keyed_key_guess == state {
            keyed_key_ok += 1;
        }
    }

    CryptResult {
        substitution_acc: sub_ok as f64 / test_msgs as f64,
        keyed_acc: keyed_ok as f64 / test_msgs as f64,
        keyed_with_key_acc: keyed_key_ok as f64 / test_msgs as f64,
    }
}

/// Result of signing a GARY artifact with ML-DSA (FIPS-204): the genuine signature must
/// verify, and any tampering of the artifact or the signature must be rejected.
pub struct PqcDemo {
    pub valid_ok: bool,
    pub tampered_msg_ok: bool,
    pub tampered_sig_ok: bool,
    pub sig_len: usize,
}

/// Sign a GARY result artifact with ML-DSA-65 and check its provenance guarantees. A
/// post-quantum signature makes published results integrity-checkable and tamper-evident.
pub fn pqc_sign_artifact(artifact: &[u8]) -> Result<PqcDemo, &'static str> {
    let (pk, sk) = ml_dsa_65::try_keygen()?;
    let ctx = b"GARY/result/v1";
    let sig = sk.try_sign(artifact, ctx)?;
    let valid_ok = pk.verify(artifact, &sig, ctx);

    // Falsifier 1: a tampered artifact must NOT verify under the genuine signature.
    let mut bad_msg = artifact.to_vec();
    if let Some(b) = bad_msg.first_mut() {
        *b ^= 0x01;
    }
    let tampered_msg_ok = pk.verify(&bad_msg, &sig, ctx);

    // Falsifier 2: a tampered signature must NOT verify.
    let mut bad_sig = sig;
    bad_sig[0] ^= 0x01;
    let tampered_sig_ok = pk.verify(artifact, &bad_sig, ctx);

    Ok(PqcDemo {
        valid_ok,
        tampered_msg_ok,
        tampered_sig_ok,
        sig_len: sig.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitution_breaks_keyed_holds() {
        let r = cryptanalysis(16, 20000, 20000, 0.5, 7);
        assert!(r.substitution_acc > 0.8, "substitution recovered from cribs");
        assert!(r.keyed_acc < 0.15, "keyed stays near chance (1/16) without the key");
        assert!(r.keyed_with_key_acc > 0.99, "keyed decodes perfectly with the key");
    }

    #[test]
    fn pqc_signs_and_rejects_tampering() {
        let art = b"GARY results: MI=3.000, CHSH=0.8536, QMI(Bell)=2.0";
        let d = pqc_sign_artifact(art).expect("keygen/sign");
        assert!(d.valid_ok, "genuine signature verifies");
        assert!(!d.tampered_msg_ok, "tampered artifact rejected");
        assert!(!d.tampered_sig_ok, "tampered signature rejected");
    }
}
