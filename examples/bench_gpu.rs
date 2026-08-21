use luckfind::gpu::{self, GpuScanner};
fn main() -> anyhow::Result<()> {
    let ctx = gpu::GpuContext::new_blocking(0)?;
    let candidates = vec![[0u32; 5]; 78];
    let mut scanner = GpuScanner::new(ctx, &candidates)?;
    scanner.init_random(luckfind::puzzles::puzzle_set())?;
    scanner.steps_per_call = 100;
    let t0 = std::time::Instant::now();
    for _ in 0..30 { scanner.step()?; }
    let elapsed = t0.elapsed();
    let total_keys = 30u64 * 100 * 100_000;
    println!("Elapsed:    {:.2?}", elapsed);
    println!("Rate:       {:.2} Mkeys/s", total_keys as f64 / elapsed.as_secs_f64() / 1e6);
    Ok(())
}
