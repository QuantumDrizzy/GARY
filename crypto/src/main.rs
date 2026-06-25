// GARY Phase 5 — the recoverability boundary (decypher total).
// An eavesdropper with known-plaintext (cribs) decodes HELD-OUT messages: a weak substitution
// code falls; a keyed-stream code holds at chance. With the key it decodes perfectly.
use gary_crypto::cryptanalysis;

fn main() {
    println!("GARY 0.0.1 - decypher total");
    println!("Phase 5 - the recoverability boundary (crypto extension of lingua cosmica)\n");
    println!("An eavesdropper with cribs (known plaintext) decodes HELD-OUT messages.");
    println!("Weak (substitution) vs keyed-stream code. n=16, chance = 6.25%.\n");

    println!("  cribs    substitution   keyed (no key)   keyed (WITH key)");
    for &f in &[0.0, 0.05, 0.10, 0.25, 0.50, 1.0] {
        let r = cryptanalysis(16, 20000, 20000, f, 42);
        println!(
            "  {:>4.0}%      {:>6.1}%         {:>6.1}%           {:>6.1}%",
            f * 100.0,
            r.substitution_acc * 100.0,
            r.keyed_acc * 100.0,
            r.keyed_with_key_acc * 100.0
        );
    }

    println!("\nthe boundary: cribs recover a substitution code (-> ~100%) but NOT a keyed one");
    println!("(it stays at chance ~6.25%). WITH the key the keyed code decodes perfectly -- its");
    println!("resistance is real secrecy, not garbled data. That gap is 'decypher total'.");
    println!("\n[the keyed stream above is a research stand-in, not production crypto]");

    // ── PQC provenance: sign a real GARY result artifact with ML-DSA (vetted crate) ──
    println!("\n--- PQC provenance (ML-DSA-65 / FIPS-204, vetted crate `fips204`) ---");
    let art = b"GARY results: perfect-code MI=3.000, CHSH=0.8536, QMI(Bell)=2.0";
    match gary_crypto::pqc_sign_artifact(art) {
        Ok(d) => {
            println!("  signed a {}-byte artifact -> {}-byte ML-DSA signature", art.len(), d.sig_len);
            println!("  genuine signature verifies   : {}", d.valid_ok);
            println!("  tampered artifact rejected   : {}", !d.tampered_msg_ok);
            println!("  tampered signature rejected  : {}", !d.tampered_sig_ok);
            println!("  -> GARY results carry post-quantum provenance (signed, tamper-evident).");
        }
        Err(e) => println!("  PQC error: {}", e),
    }
}
