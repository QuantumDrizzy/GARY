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
    println!("\n[research models, not production crypto; real PQC pillar = ML-DSA signing, next]");
}
