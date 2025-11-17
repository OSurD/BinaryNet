#include <torch/extension.h>
#include <cstdint>

static inline std::uint32_t popcnt_u64(std::uint64_t v) {
#ifdef _MSC_VER
    return static_cast<std::uint32_t>(__popcnt64(v));
#else
    return static_cast<std::uint32_t>(__builtin_popcountll(static_cast<unsigned long long>(v)));
#endif
}

torch::Tensor pack_row(const torch::Tensor& A)
{
    TORCH_CHECK(A.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat);

    auto A_contig = A.contiguous();

    const int64_t M = A_contig.size(0);
    const int64_t K = A_contig.size(1);
    const int64_t nblocks = (K + 63) / 64;

    auto opts = torch::TensorOptions()
                    .dtype(torch::kLong)
                    .device(torch::kCPU);

    auto packed = torch::zeros({M, nblocks}, opts);

    const float* A_ptr = A_contig.data_ptr<float>();
    int64_t* P_ptr = packed.data_ptr<int64_t>();

    for (int64_t i = 0; i < M; ++i) {
        const float* rowA = A_ptr + i * K;
        int64_t* rowP = P_ptr + i * nblocks;

        for (int64_t k = 0; k < K; ++k) {
            const int64_t blk = k >> 6; //< k / 64
            const int64_t off = k & 63; //< k % 64

            bool bit = (rowA[k] > 0);
            rowP[blk] |= (static_cast<int64_t>(bit) << off);
        }
    }

    return packed;
}

torch::Tensor pack_col(const torch::Tensor& B) {
    TORCH_CHECK(B.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat);

    auto B_contig = B.contiguous();

    const int64_t K = B_contig.size(0);
    const int64_t N = B_contig.size(1);
    const int64_t nblocks = (K + 63) / 64;

    auto opts = torch::TensorOptions()
                    .dtype(torch::kLong)
                    .device(torch::kCPU);

    auto packed = torch::zeros({N, nblocks}, opts);

    const float*  B_ptr = B_contig.data_ptr<float>();
    int64_t* P_ptr = packed.data_ptr<int64_t>();

    for (int64_t j = 0; j < N; ++j) {
        int64_t* rowP = P_ptr + j * nblocks;

        for (int64_t k = 0; k < K; ++k) {
            const int64_t blk = k >> 6; //< k / 64
            const int64_t off = k & 63; //< k % 64

            bool bit = (B_ptr[k * N + j] > 0);
            rowP[blk] |= (static_cast<int64_t>(bit) << off);
        }
    }

    return packed;
}

torch::Tensor bin_matmul_xnor_popcnt_tensor(const torch::Tensor& A, const torch::Tensor& B)
{
    TORCH_CHECK(A.device().is_cpu() && B.device().is_cpu(), "CPU only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expected 2D tensors");
    TORCH_CHECK(
        (A.scalar_type() == torch::kFloat) &&
        (B.scalar_type() == torch::kFloat)
    );

    const int64_t M  = A.size(0);
    const int64_t K  = A.size(1);
    const int64_t K2 = B.size(0);
    const int64_t N  = B.size(1);

    TORCH_CHECK(K == K2, "inner dimensions must match");

    auto A_packed = pack_row(A);   // int64, [M, nblocks]
    auto B_packed = pack_col(B);   // int64, [N, nblocks]

    A_packed = A_packed.contiguous();
    B_packed = B_packed.contiguous();

    const int64_t nb  = A_packed.size(1);
    TORCH_CHECK(A_packed.size(0) == M);
    TORCH_CHECK(B_packed.size(0) == N);
    TORCH_CHECK(B_packed.size(1) == nb, "inconsistent nblocks between A and B");

    auto out_opts = torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(torch::kCPU);

    auto C = torch::empty({M, N}, out_opts);

    // маска для последнего блока
    const int64_t valid_tail = (K & 63);
    const std::uint64_t tail_mask =
        (valid_tail == 0)
        ? ~std::uint64_t{0}                        //< все биты валидны
        : ((std::uint64_t{1} << valid_tail) - 1u); //< только K%64 младших битов

    const int64_t* A_ptr = A_packed.data_ptr<int64_t>();
    const int64_t* B_ptr = B_packed.data_ptr<int64_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();

    for (int64_t i = 0; i < M; ++i) {
        const int64_t* rowA = A_ptr + i * nb;
        int32_t* rowC = C_ptr + i * N;

        for (int64_t j = 0; j < N; ++j) {
            const int64_t* rowB = B_ptr + j * nb;

            std::uint32_t equal_bits = 0;

            for (int64_t blk = 0; blk < nb; ++blk) {
                std::uint64_t a = static_cast<std::uint64_t>(rowA[blk]);
                std::uint64_t b = static_cast<std::uint64_t>(rowB[blk]);

                std::uint64_t x = ~(a ^ b); //< XNOR

                if (blk == nb - 1) {
                    x &= tail_mask;
                }

                equal_bits += popcnt_u64(x);
            }

            // Т.к. входы {+1,-1}, то dot = 2 * (equal_bits) - K
            const int32_t dot = 2 * static_cast<int32_t>(equal_bits)
                                     - static_cast<int32_t>(K);

            rowC[j] = dot;
        }
    }
    return C;
}


TORCH_LIBRARY(my_op_2, m)
{
    m.def("bin_matmul_xnor_popcnt_tensor", &bin_matmul_xnor_popcnt_tensor);
}