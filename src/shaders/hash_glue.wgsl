// =============================================================================
// Hash glue: adapt the kangaroo sha256/ripemd160 process_block primitives to the
// sha256_block / ripemd160_block interface that luckfind.wgsl already calls.
//
// luckfind.wgsl hands us a fully-padded 16-word block (pubkey_to_sha256_block /
// sha256_to_ripemd160_build already did the padding), so we just initialize the
// IVs and run one block through the upstream compressor — no re-padding here.
// =============================================================================

fn sha256_block(block: array<u32, 16>) -> array<u32, 8> {
    var state = array<u32, 8>(H0, H1, H2, H3, H4, H5, H6, H7);
    sha256_process_block(&state, block);
    return state;
}

fn ripemd160_block(block: array<u32, 16>) -> array<u32, 5> {
    var state = array<u32, 5>(RH0, RH1, RH2, RH3, RH4);
    rmd160_process_block(&state, block);
    return state;
}
