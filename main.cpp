#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>
#include <string.h>


/*
#define N 16384
//#define N 32768
//#define N 131072

void init_f(double* p, int power, bool powerOf2) {
    for (int i = 0; i < N; i++) p[i] = 0.0;
    if (powerOf2) {
        p[0] = 1.0;
        p[power / 2] = 1.0;
    }
    else {
        for (int i = 0; i < N; i++) p[i] = (i < power) ? 1.0 : 0.0;
    }
}

void multiply_polys(double* a, double* b, double* result) {
    fftw_complex* A = fftw_alloc_complex(N);
    fftw_complex* B = fftw_alloc_complex(N);
    fftw_complex* C = fftw_alloc_complex(N);

    fftw_plan p1 = fftw_plan_dft_r2c_1d(N, a, A, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_r2c_1d(N, b, B, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_execute(p2);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);

    for (int i = 0; i < N; i++) {
        double reA = A[i][0], imA = A[i][1];
        double reB = B[i][0], imB = B[i][1];
        C[i][0] = reA * reB - imA * imB;
        C[i][1] = reA * imB + imA * reB;
    }

    fftw_plan pinv = fftw_plan_dft_c2r_1d(N, C, result, FFTW_ESTIMATE);
    fftw_execute(pinv);
    fftw_destroy_plan(pinv);
    for (int i = 0; i < N; i++) result[i] /= N;

    fftw_free(A);
    fftw_free(B);
    fftw_free(C);
    fftw_cleanup();
}

// f(x) = f1 * f2 * f3
void compute_f(double* f, int q0, int q1, int q2) {
    double* f1 = fftw_alloc_real(N);
    double* f2 = fftw_alloc_real(N);
    double* f3 = fftw_alloc_real(N);
    fftw_complex* F1 = fftw_alloc_complex(N);
    fftw_complex* F2 = fftw_alloc_complex(N);
    fftw_complex* F3 = fftw_alloc_complex(N);
    fftw_complex* F = fftw_alloc_complex(N);

    init_f(f1, q0, true);
    init_f(f2, q1, false);
    init_f(f3, q2, false);

    fftw_plan p1 = fftw_plan_dft_r2c_1d(N, f1, F1, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_r2c_1d(N, f2, F2, FFTW_ESTIMATE);
    fftw_plan p3 = fftw_plan_dft_r2c_1d(N, f3, F3, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_execute(p2);
    fftw_execute(p3);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);

    for (int i = 0; i < N; i++) {
        double re1 = F1[i][0], im1 = F1[i][1];
        double re2 = F2[i][0], im2 = F2[i][1];
        double re3 = F3[i][0], im3 = F3[i][1];
        double temp_re, temp_im;

        temp_re = re1 * re2 - im1 * im2;
        temp_im = re1 * im2 + im1 * re2;
        F[i][0] = temp_re * re3 - temp_im * im3;
        F[i][1] = temp_re * im3 + temp_im * re3;
    }

    fftw_plan pinv = fftw_plan_dft_c2r_1d(N, F, f, FFTW_ESTIMATE);
    fftw_execute(pinv);
    fftw_destroy_plan(pinv);
    for (int i = 0; i < N; i++) f[i] /= N;

    fftw_free(f1);
    fftw_free(f2);
    fftw_free(f3);
    fftw_free(F1);
    fftw_free(F2);
    fftw_free(F3);
    fftw_free(F);
    fftw_cleanup();
}

void poly_mod(double* a, double* f, int n) {
    int deg_f = n - 1;
    while (deg_f >= 0 && fabs(f[deg_f]) < 1e-10) deg_f--;

    if (deg_f < 0) {
        printf("Illegal modular polynomial\n");
        exit(1);
    }

    while (1) {
        int deg_a = n - 1;
        while (deg_a >= 0 && fabs(a[deg_a]) < 1e-10) deg_a--;
        if (deg_a < deg_f) break;

        double factor = a[deg_a] / f[deg_f];
        int shift = deg_a - deg_f;

        for (int i = 0; i <= deg_f; i++) {
            a[shift + i] -= factor * f[i];
        }
    }
}

void poly_mul(double* a, double* b, double* c, int n) {

    for (int i = 0; i < 2 * n - 1; i++) c[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i + j] += a[i] * b[j];
        }
    }
}

void poly_mod2(double* c, double* f, int max_deg_c, int deg_f) {

    for (int i = max_deg_c; i >= deg_f; i--) {
        if (fabs(c[i]) < 1e-10) continue;

        double factor = c[i] / f[deg_f];
        for (int j = 0; j <= deg_f; j++) {
            c[i - j] -= factor * f[deg_f - j];
        }
    }
}

void random_poly(double* p, int n, int max_coeff, int density) {
    for (int i = 0; i < n; i++) {
        if (rand() % density == 0) {
            p[i] = (double)(rand() % max_coeff + 1); 
        }
        else {
            p[i] = 0.0;
        }
    }
}

int read_poly_to_fftw(const char* filename, double* f, int max_len) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Cannot open the file");
        return -1;
    }

    char* line = (char*)malloc(1024 * 1024);  // 1MB 缓冲区
    if (!line) {
        printf("Memory allocation failed\n");
        fclose(fp);
        return -1;
    }

    if (fgets(line, 1024 * 1024, fp) == NULL) {
        printf("The file is empty or the read failed\n");
        free(line);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    int count = 0;
    char* token = strtok(line, ", ");
    while (token != NULL && count < max_len) {
        f[count++] = atof(token);
        token = strtok(NULL, ", ");
    }
    free(line);
    return count;
}

int main() {
    //int q0 = 4, q1 = 17, q2 = 241;  const char* filename = "16388.txt";  //#define N 32768
    //int q0 = 4, q1 = 17, q2 = 433;  const char* filename = "29444.txt";  //#define N 32768
    int q0 = 4, q1 = 41, q2 = 73;  const char* filename = "11972.txt";  //#define N 16384
    //int q0 = 4, q1 = 73, q2 = 97;  const char* filename = "28324.txt";  //#define N 32768
    //int q0 = 4, q1 = 97, q2 = 193;  const char* filename = "74884.txt";  //#define N 131072



    int q = q0 * q1 * q2;
    double* a = fftw_alloc_real(N);
    double* b = fftw_alloc_real(N);
    double* c = fftw_alloc_real(N);
    double* d = fftw_alloc_real(2 * N);
    double* f = fftw_alloc_real(N);

    random_poly(a, int(q/2), 10, 3);
    random_poly(b, int(q/2), 10, 3);

    // Step 1: f(x)
    int num_read = read_poly_to_fftw(filename, f, N);
    //compute_f(f, q0, q1, q2);


    // Step 2: c = fftw(a * b)
    multiply_polys(a, b, c);

    // Step 3: c mod f(x)
    poly_mod(c, f, N);

    // Step 4: d = a * b
    poly_mul(a, b, d, N);

    // Step 5: d mod f(x)
    poly_mod2(d, f, 2 * N - 1, num_read - 1);

    // Step 6: Verify
    printf("Coefficients of two results:\n");
    for (int i = 0; i < N; i++) {
        if (fabs(c[i]) > 1e-10) {
            printf("x^%d fftw: %.6f, mult: %.6f", i, c[i], d[i]);
            if (unsigned int (c[i]-d[i]) < 1e-6) printf("   equal\n");
            else printf("    not equal\n");
        }
    }

    // Clear
    fftw_free(a);
    fftw_free(b);
    fftw_free(c);
    fftw_free(d);
    fftw_free(f);
    fftw_cleanup();
    return 0;
}
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef int elem;

typedef struct {
    int n;              // total length (power of 2)
    int k;              // block size (power of 2)
    int num_blocks;     // n / k
    int pnt_levels;     // log2(num_blocks)
    elem q;         // prime modulus, q ≡ 1 (mod 2n)
    elem* zeta;     // precomputed roots: zeta[i] = ω^i mod q
} pnt_context_t;


// Modular multiplication
static inline elem mul_mod(elem a, elem b, elem q) {
    return (a * b) % q;
}

// Fast modular exponentiation
static elem pow_mod(elem base, elem exp, elem q) {
    elem res = 1;
    base %= q;
    while (exp) {
        if (exp & 1) res = mul_mod(res, base, q);
        base = mul_mod(base, base, q);
        exp >>= 1;
    }
    return res;
}

// Bit-reversal permutation
static void bit_reverse(elem* a, int n) {
    if (n <= 2) return;
    int j = 0;
    for (int i = 1; i < n - 1; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if (i < j) {
            elem t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}

// Compute log2(x) for x = 2^k
static int log2_int(int x) {
    if (x <= 0) return -1;
    int log = 0;
    while (x > 1) {
        log++;
        x >>= 1;
    }
    return log;
}

// Check if x is power of two
static int is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

pnt_context_t* pnt_init(int n, int k, elem q, elem g);

int pnt_transform(const pnt_context_t* ctx, const elem* poly, elem* poly2);//0 suc 1 failed

int ipnt_transform(const pnt_context_t* ctx, const elem* poly2, elem* poly3);//0 suc 1 failed

void pnt_free(pnt_context_t* ctx);



pnt_context_t* pnt_init(int n, int k, elem q, elem g) {
    // Validate parameters
    if (!is_power_of_two(n)) return NULL;
    if (!is_power_of_two(k)) return NULL;
    if (k > n || n % k != 0) return NULL;
    if (q == 0 || (q - 1) % (2 * (elem)n) != 0) return NULL;

    pnt_context_t* ctx = (pnt_context_t*)malloc(sizeof(pnt_context_t));
    if (!ctx) return NULL;

    ctx->n = n;
    ctx->k = k;
    ctx->num_blocks = n / k;
    ctx->pnt_levels = log2_int(ctx->num_blocks);
    ctx->q = q;

    ctx->zeta = (elem*)malloc(n * sizeof(elem));
    if (!ctx->zeta) {
        free(ctx);
        return NULL;
    }

    // Find primitive root: use g=3 (works for many small primes)
    //elem exponent =  (q - 1) / (2 * (elem)n);
    //elem omega = pow_mod(g, exponent, q);  // ω = g^((q-1)/(2n)) mod q

    elem omega = g;

    // Precompute zeta table: zeta[i] = ω^i mod q
    ctx->zeta[0] = 1;
    for (int i = 1; i < n; i++) {
        ctx->zeta[i] = mul_mod(ctx->zeta[i - 1], omega, q);
    }

    return ctx;
}

int pnt_transform(const pnt_context_t* ctx, const elem* poly, elem* poly2) {
    if (!ctx || !poly || !poly2) return -1;

    const int n = ctx->n;
    const elem q = ctx->q;

    // Copy input and apply bit-reversal
    for (int i = 0; i < n; i++) {
        poly2[i] = poly[i];
    }
    //bit_reverse(poly2, n);

    // Apply first L = log2(n/k) layers of NTT
    for (int level = 0; level < ctx->pnt_levels; level++) {
        int width = 1 << (level + 1);      // butterfly width
        int groups = n / width;           // number of groups
        int step = n >> (level + 1);      // root step
        elem w = ctx->zeta[step];     // ω^step

        for (int g = 0; g < groups; g++) {
            elem wk = 1;
            int base = g * width;
            for (int j = 0; j < width / 2; j++) {
                elem t = mul_mod(wk, poly2[base + j + width / 2], q);
                elem u = poly2[base + j];
                poly2[base + j] = (u + t) % q;
                poly2[base + j + width / 2] = (u + q - t) % q;
                wk = mul_mod(wk, w, q);
            }
        }
    }

    return 0;
}

int ipnt_transform(const pnt_context_t* ctx, const elem* poly2, elem* poly3) {
    if (!ctx || !poly2 || !poly3) return -1;

    const int n = ctx->n;
    const elem q = ctx->q;
    const int num_blocks = ctx->num_blocks;

    // Copy input
    for (int i = 0; i < n; i++) {
        poly3[i] = poly2[i];
    }

    // Inverse butterfly layers (reverse order)
    for (int level = ctx->pnt_levels - 1; level >= 0; level--) {
        int width = 1 << (level + 1);
        int groups = n / width;
        int step = n >> (level + 1);
        elem w = ctx->zeta[step];
        elem inv_w = pow_mod(w, q - 2, q);  // inverse by Fermat

        for (int g = 0; g < groups; g++) {
            elem w_inv_k = 1;
            int base = g * width;
            for (int j = 0; j < width / 2; j++) {
                elem u = poly3[base + j];
                elem v = poly3[base + j + width / 2];

                elem x = (u + v) % q;
                elem y = (u + q - v) % q;
                y = mul_mod(y, w_inv_k, q);

                poly3[base + j] = x;
                poly3[base + j + width / 2] = y;

                w_inv_k = mul_mod(w_inv_k, inv_w, q);
            }
        }
    }

    // Final bit-reversal
    //bit_reverse(poly3, n);

    // Scale by 1/(n/k) = k/n
    elem inv_scale = pow_mod(num_blocks, q - 2, q);
    for (int i = 0; i < n; i++) {
        poly3[i] = mul_mod(poly3[i], inv_scale, q);
    }

    return 0;
}

void pnt_free(pnt_context_t* ctx) {
    if (ctx) {
        if (ctx->zeta) {
            free(ctx->zeta);
        }
        free(ctx);
    }
}

// ------------------------
// Example Usage (uncomment to test)
// ------------------------


int main() {
    int n = 8;
    int k = 4;
    elem q =  17*257;
    elem g = 260;
    //17  5
    //257  3
    //17*257  260
    //241 7
    //17*241 1212
    pnt_context_t *ctx = pnt_init(n, k, q, g);
    if (!ctx) {
        printf("Error: pnt_init failed.\n");
        return 1;
    }

    elem poly[32]   = {0};
    for (int i = 0; i < n; i++) {
        poly[i] = i;
    }
    elem poly2[32]; // PNT output
    elem poly3[32]; // iPNT output

    printf("Original: ");
    for (int i = 0; i < n; i++) printf("%lu ", poly[i]); printf("\n");

    pnt_transform(ctx, poly, poly2);
    printf("PNT:      ");
    for (int i = 0; i < n; i++) printf("%lu ", poly2[i]); printf("\n");

    ipnt_transform(ctx, poly2, poly3);
    printf("iPNT:     ");
    for (int i = 0; i < n; i++) printf("%lu ", poly3[i]); printf("\n");

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (poly3[i] != poly[i]) {
            ok = 0; break;
        }
    }
    printf("Recovery: %s\n", ok ? "SUCCESS" : "FAILED");

    pnt_free(ctx);
    return 0;
}









