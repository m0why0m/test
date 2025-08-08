use tfhe::core_crypto::prelude::*;

fn main() {

    // parameters
    let lwe_dimension = LweDimension(760);
    let glwe_dimension = GlweDimension(1);
    let polynomial_size = PolynomialSize(1024);
    let lwe_modular_std_dev = StandardDev(0.0000000000000002940360153546159);
    let glwe_modular_std_dev = StandardDev(0.0000000000000002940360153546159);
    let pbs_level = DecompositionLevelCount(3);
    let pbs_base_log = DecompositionBaseLog(7);
    let cbs_level = DecompositionLevelCount(5);
    let cbs_base_log = DecompositionBaseLog(3);
    let ciphertext_modulus = CiphertextModulus::new_native();

    let mut seeder = new_seeder();
    let seeder = seeder.as_mut();
    let mut encryption_generator = EncryptionRandomGenerator::new(seeder.seed(), seeder);
    let mut secret_generator = SecretRandomGenerator::new(seeder.seed());

    // lwe key, glwe key, ggsw key
    let lwe_sk = allocate_and_generate_new_binary_lwe_secret_key(
        lwe_dimension, &mut secret_generator,
    );

    let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
        glwe_dimension, polynomial_size, &mut secret_generator,
    );

    let ggsw_sk = glwe_sk.clone();


    // Bootstrap Key,  PF-LUT Key
    let bsk = allocate_and_generate_new_lwe_bootstrap_key(
        &lwe_sk,
        &glwe_sk,
        pbs_level,
        pbs_base_log,
        glwe_modular_std_dev,
        ciphertext_modulus,
        &mut encryption_generator,
    );

    let pfpksk_list = allocate_and_generate_new_circuit_bootstrap_lwe_pfpksk_list(
        &lwe_sk,
        &ggsw_sk,
        cbs_level,
        cbs_base_log,
        glwe_modular_std_dev,
        polynomial_size,
        ciphertext_modulus,
        &mut encryption_generator,
    );

    // transfrom bsk to fft
    let mut fourier_bsk = FourierLweBootstrapKey::new(
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        pbs_level,
        pbs_base_log,
    );

    convert_standard_lwe_bootstrap_key_to_fourier(&bsk, &mut fourier_bsk);

    // generate lut
    let delta: u64 = 1 << 63;

    // generate a polynomial for storing lut, f(x) = NOT(x)
    let mut lut_poly = vec![0u64; polynomial_size.0];
    for i in 0..polynomial_size.0 {
        // first half x=1, output 0，second half x=0 output delta
        lut_poly[i] = if i < polynomial_size.0 / 2 { 0 } else { delta };
    }

    let big_lut_as_polynomial_list = PlaintextList::from_container(lut_poly)
        .into_polynomial_list();

    // input/output lwe ciphertext
    let clear_bit = 1u64;
    let plaintext = Plaintext(clear_bit << 63);

    let mut lwe_in = LweCiphertext::new(0u64, lwe_dimension.to_lwe_size(), ciphertext_modulus);
    encrypt_lwe_ciphertext(
        &lwe_sk,
        &mut lwe_in,
        &plaintext,
        lwe_modular_std_dev,
        &mut encryption_generator,
    );

    let mut lwe_out = LweCiphertext::new(0u64, lwe_dimension.to_lwe_size(), ciphertext_modulus);

    // pack to list
    let lwe_list_in = LweCiphertextList::from_container(
        lwe_in.as_container(),
        lwe_dimension.to_lwe_size(),
        CiphertextModulus::new_native(),
    );

    let mut lwe_list_out = LweCiphertextList::new(
        0u64,
        lwe_dimension.to_lwe_size(),
        1, // number of ciphertext
        ciphertext_modulus,
    );

    // fft
    let mut fft = Fft::new(polynomial_size);
    let mut fft_cache = FftCache::new();
    let fft = fft.as_view();

    // cal memory
    let stack_size = circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_list_mem_optimized::stack_size_required(
        lwe_list_in.lwe_ciphertext_count(),
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        pbs_level,
    );

    let mut stack_vec = vec![0u64; stack_size];
    let mut stack = PodStack::new(&mut stack_vec);

    // circuit bootstrapping
    circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_list_mem_optimized(
        &lwe_list_in,
        &mut lwe_list_out,
        &big_lut_as_polynomial_list,
        &fourier_bsk,
        &pfpksk_list,
        cbs_base_log,
        cbs_level,
        fft,
        &mut stack,
    );

    // dec
    let lwe_out = lwe_list_out.get_lwe_ciphertext(0);
    let decrypted = decrypt_lwe_ciphertext(&lwe_sk, lwe_out);
    let result_bit = (decrypted.0 >> 63) & 1;

    println!("input: {}, output: {}", clear_bit, result_bit); // 1 -> 0
}