#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nevrine/core/arena.hpp"
#include "nevrine/core/cuda_utils.hpp"
#include "nevrine/core/device_info.hpp"
#include "nevrine/cpu/activations_cpu.hpp"
#include "nevrine/cpu/conv2d_cpu.hpp"
#include "nevrine/cpu/gemm_cpu.hpp"
#include "nevrine/cuda/activations.hpp"
#include "nevrine/cuda/conv2d.hpp"
#include "nevrine/cuda/cublas_gemm.hpp"
#include "nevrine/cuda/gemm_naive.hpp"
#include "nevrine/cuda/gemm_tiled.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace nevrine;

namespace {

// Thread-local or lazily initialized DeviceMemoryArena for Python ops
core::DeviceMemoryArena& getInternalArena() {
    // 256 MB reusable default scratchpad arena
    static thread_local core::DeviceMemoryArena default_arena(256 * 1024 * 1024, 256);
    return default_arena;
}

// Global cuBLAS handle wrapper for Python bindings
cuda::CublasHandle& getCublasHandle() {
    static cuda::CublasHandle handle;
    return handle;
}

// Helper to ensure float32 c-contiguous array
py::array_t<float, py::array::c_style | py::array::forcecast> ensureContiguousFloat(
    const py::array_t<float>& arr, const std::string& name) {
    if (arr.ndim() == 0) {
        throw std::invalid_argument(name + " must not be a 0-dimensional scalar");
    }
    return py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(arr);
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Operations Implementations
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_gemm(py::array_t<float> a_arr,
                           py::array_t<float> b_arr,
                           float alpha,
                           float beta,
                           const std::string& backend) {
    auto a = ensureContiguousFloat(a_arr, "Matrix A");
    auto b = ensureContiguousFloat(b_arr, "Matrix B");

    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("GEMM inputs A and B must both be 2D matrices");
    }

    int M = static_cast<int>(a.shape(0));
    int K_a = static_cast<int>(a.shape(1));
    int K_b = static_cast<int>(b.shape(0));
    int N = static_cast<int>(b.shape(1));

    if (K_a != K_b) {
        throw std::invalid_argument("Inner dimensions do not match: A is (" +
                                    std::to_string(M) + ", " + std::to_string(K_a) +
                                    ") but B is (" + std::to_string(K_b) + ", " +
                                    std::to_string(N) + ")");
    }
    int K = K_a;

    // Allocate output array (M x N)
    py::array_t<float> c_arr({M, N});
    float* c_ptr = c_arr.mutable_data();
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();

    if (backend == "cpu") {
        std::fill(c_ptr, c_ptr + (M * N), 0.0f);
        cpu::gemmCpuReference(a_ptr, b_ptr, c_ptr, M, N, K, alpha, beta);
        return c_arr;
    }

    // GPU path
    size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);
    size_t total_needed = bytes_a + bytes_b + bytes_c + 8192;

    auto& arena = getInternalArena();
    if (total_needed > arena.capacity()) {
        throw core::ArenaOutOfMemoryException(total_needed, arena.remaining_bytes(), arena.capacity());
    }

    arena.reset();
    float* d_a = static_cast<float*>(arena.allocate(bytes_a));
    float* d_b = static_cast<float*>(arena.allocate(bytes_b));
    float* d_c = static_cast<float*>(arena.allocate(bytes_c));

    CUDA_CHECK(cudaMemcpy(d_a, a_ptr, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_ptr, bytes_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));

    if (backend == "cuda_tiled" || backend == "cuda" || backend == "default") {
        cuda::launchTiledGemm(d_a, d_b, d_c, M, N, K, alpha, beta);
    } else if (backend == "cuda_naive") {
        cuda::launchNaiveGemm(d_a, d_b, d_c, M, N, K, alpha, beta);
    } else if (backend == "cublas") {
        cuda::launchCublasGemm(getCublasHandle().get(), d_a, d_b, d_c, M, N, K, alpha, beta);
    } else {
        throw std::invalid_argument("Unknown backend '" + backend +
                                    "'. Supported: 'cuda_tiled', 'cuda_naive', 'cublas', 'cpu'");
    }

    CUDA_SYNC_AND_CHECK();
    CUDA_CHECK(cudaMemcpy(c_ptr, d_c, bytes_c, cudaMemcpyDeviceToHost));
    arena.reset();

    return c_arr;
}

py::array_t<float> py_conv2d(py::array_t<float> input_arr,
                             py::array_t<float> kernel_arr,
                             std::optional<py::array_t<float>> bias_arr,
                             int stride,
                             int pad,
                             bool fuse_relu,
                             const std::string& backend) {
    auto input = ensureContiguousFloat(input_arr, "Input");
    auto kernel = ensureContiguousFloat(kernel_arr, "Kernel");

    if (input.ndim() != 4) {
        throw std::invalid_argument("Input tensor must be 4D (N, C_in, H, W)");
    }
    if (kernel.ndim() != 4) {
        throw std::invalid_argument("Kernel tensor must be 4D (C_out, C_in, kH, kW)");
    }

    int N = static_cast<int>(input.shape(0));
    int C_in = static_cast<int>(input.shape(1));
    int H = static_cast<int>(input.shape(2));
    int W = static_cast<int>(input.shape(3));

    int C_out = static_cast<int>(kernel.shape(0));
    int k_Cin = static_cast<int>(kernel.shape(1));
    int kH = static_cast<int>(kernel.shape(2));
    int kW = static_cast<int>(kernel.shape(3));

    if (C_in != k_Cin) {
        throw std::invalid_argument("Input channel count (" + std::to_string(C_in) +
                                    ") does not match kernel input channels (" +
                                    std::to_string(k_Cin) + ")");
    }

    const float* bias_ptr = nullptr;
    size_t bias_bytes = 0;
    py::array_t<float, py::array::c_style | py::array::forcecast> bias;
    if (bias_arr.has_value()) {
        bias = ensureContiguousFloat(bias_arr.value(), "Bias");
        if (bias.ndim() != 1 || bias.shape(0) != C_out) {
            throw std::invalid_argument("Bias must be a 1D tensor with length equal to C_out (" +
                                        std::to_string(C_out) + ")");
        }
        bias_ptr = bias.data();
        bias_bytes = static_cast<size_t>(C_out) * sizeof(float);
    }

    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("Calculated Conv2D output shape (" + std::to_string(H_out) +
                                    "x" + std::to_string(W_out) + ") is invalid.");
    }

    py::array_t<float> output_arr({N, C_out, H_out, W_out});
    float* output_ptr = output_arr.mutable_data();
    const float* input_ptr = input.data();
    const float* kernel_ptr = kernel.data();

    if (backend == "cpu") {
        cpu::conv2dCpuReference(input_ptr, kernel_ptr, bias_ptr, output_ptr,
                                N, C_in, H, W, C_out, kH, kW, stride, pad);
        if (fuse_relu) {
            size_t total_out_elems = static_cast<size_t>(N) * C_out * H_out * W_out;
            cpu::reluCpuReference(output_ptr, total_out_elems);
        }
        return output_arr;
    }

    if (backend == "cuda" || backend == "default") {
        size_t input_bytes = static_cast<size_t>(N) * C_in * H * W * sizeof(float);
        size_t kernel_bytes = static_cast<size_t>(C_out) * C_in * kH * kW * sizeof(float);
        size_t output_bytes = static_cast<size_t>(N) * C_out * H_out * W_out * sizeof(float);
        size_t total_needed = input_bytes + kernel_bytes + bias_bytes + output_bytes + 8192;

        auto& arena = getInternalArena();
        if (total_needed > arena.capacity()) {
            throw core::ArenaOutOfMemoryException(total_needed, arena.remaining_bytes(), arena.capacity());
        }

        arena.reset();
        float* d_input = static_cast<float*>(arena.allocate(input_bytes));
        float* d_kernel = static_cast<float*>(arena.allocate(kernel_bytes));
        float* d_bias = nullptr;
        if (bias_ptr != nullptr) {
            d_bias = static_cast<float*>(arena.allocate(bias_bytes));
        }
        float* d_output = static_cast<float*>(arena.allocate(output_bytes));

        CUDA_CHECK(cudaMemcpy(d_input, input_ptr, input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel_ptr, kernel_bytes, cudaMemcpyHostToDevice));
        if (bias_ptr != nullptr) {
            CUDA_CHECK(cudaMemcpy(d_bias, bias_ptr, bias_bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

        if (fuse_relu) {
            cuda::launchConv2dRelu(d_input, d_kernel, d_bias, d_output,
                                   N, C_in, H, W, C_out, kH, kW, stride, pad);
        } else {
            cuda::launchConv2d(d_input, d_kernel, d_bias, d_output,
                               N, C_in, H, W, C_out, kH, kW, stride, pad);
        }

        CUDA_SYNC_AND_CHECK();
        CUDA_CHECK(cudaMemcpy(output_ptr, d_output, output_bytes, cudaMemcpyDeviceToHost));
        arena.reset();

        return output_arr;
    }

    throw std::invalid_argument("Unknown backend '" + backend + "'. Supported: 'cuda', 'cpu'");
}

py::array_t<float> py_relu(py::array_t<float> x_arr, bool inplace, const std::string& backend) {
    auto x = ensureContiguousFloat(x_arr, "Input");
    size_t count = static_cast<size_t>(x.size());

    py::array_t<float> result;
    if (inplace) {
        result = x;
    } else {
        result = py::array_t<float>(x.request());
    }
    float* data = result.mutable_data();

    if (!inplace) {
        std::copy(x.data(), x.data() + count, data);
    }

    if (backend == "cpu") {
        cpu::reluCpuReference(data, count);
        return result;
    }

    if (backend == "cuda" || backend == "default") {
        size_t bytes = count * sizeof(float);
        auto& arena = getInternalArena();
        if (bytes + 512 > arena.capacity()) {
            throw core::ArenaOutOfMemoryException(bytes, arena.remaining_bytes(), arena.capacity());
        }

        arena.reset();
        float* d_data = static_cast<float*>(arena.allocate(bytes));
        CUDA_CHECK(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
        cuda::launchRelu(d_data, count);
        CUDA_SYNC_AND_CHECK();
        CUDA_CHECK(cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost));
        arena.reset();
        return result;
    }

    throw std::invalid_argument("Unknown backend '" + backend + "'. Supported: 'cuda', 'cpu'");
}

py::array_t<float> py_leaky_relu(py::array_t<float> x_arr,
                                 float negative_slope,
                                 bool inplace,
                                 const std::string& backend) {
    auto x = ensureContiguousFloat(x_arr, "Input");
    size_t count = static_cast<size_t>(x.size());

    py::array_t<float> result;
    if (inplace) {
        result = x;
    } else {
        result = py::array_t<float>(x.request());
    }
    float* data = result.mutable_data();

    if (!inplace) {
        std::copy(x.data(), x.data() + count, data);
    }

    if (backend == "cpu") {
        cpu::leakyReluCpuReference(data, count, negative_slope);
        return result;
    }

    if (backend == "cuda" || backend == "default") {
        size_t bytes = count * sizeof(float);
        auto& arena = getInternalArena();
        if (bytes + 512 > arena.capacity()) {
            throw core::ArenaOutOfMemoryException(bytes, arena.remaining_bytes(), arena.capacity());
        }

        arena.reset();
        float* d_data = static_cast<float*>(arena.allocate(bytes));
        CUDA_CHECK(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
        cuda::launchLeakyRelu(d_data, count, negative_slope);
        CUDA_SYNC_AND_CHECK();
        CUDA_CHECK(cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost));
        arena.reset();
        return result;
    }

    throw std::invalid_argument("Unknown backend '" + backend + "'. Supported: 'cuda', 'cpu'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Pybind11 Module Definition
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_nevrine, m) {
    m.doc() = "Nevrine Cloud-Native Neural Acceleration Engine Python Bindings";

    // Device Manager & Info
    py::class_<core::DeviceManager>(m, "DeviceManager")
        .def_static("get_device_count", &core::DeviceManager::getDeviceCount,
                    "Returns total number of CUDA-capable devices available.")
        .def_static("get_current_device", &core::DeviceManager::getCurrentDevice,
                    "Returns active CUDA device ID.")
        .def_static("set_device", &core::DeviceManager::setDevice, py::arg("device_id"),
                    "Sets active CUDA device ID.")
        .def_static("get_memory_info", [](int device_id) {
            core::DeviceManager::setDevice(device_id);
            auto mem = core::DeviceManager::getMemoryInfo();
            py::dict d;
            d["free_bytes"] = mem.free_bytes;
            d["total_bytes"] = mem.total_bytes;
            d["used_bytes"] = mem.used_bytes;
            d["free_mib"] = mem.free_mib();
            d["total_mib"] = mem.total_mib();
            d["used_mib"] = mem.used_mib();
            return d;
        }, py::arg("device_id") = 0, "Returns VRAM usage metrics dictionary.")
        .def_static("get_device_properties", [](int device_id) {
            auto props = core::DeviceManager::getDeviceProperties(device_id);
            py::dict d;
            d["device_id"] = props.device_id;
            d["name"] = props.name;
            d["compute_capability"] = std::to_string(props.major) + "." + std::to_string(props.minor);
            d["total_global_mem_bytes"] = props.total_global_mem;
            d["multi_processor_count"] = props.multi_processor_count;
            d["max_threads_per_block"] = props.max_threads_per_block;
            d["warp_size"] = props.warp_size;
            d["shared_mem_per_block_bytes"] = props.shared_mem_per_block;
            d["l2_cache_size_bytes"] = props.l2_cache_size;
            return d;
        }, py::arg("device_id") = 0, "Returns hardware device properties dictionary.")
        .def_static("print_device_summary", &core::DeviceManager::printDeviceSummary,
                    py::arg("device_id") = 0, "Prints hardware specification summary to stdout.");

    // Device Memory Arena
    py::class_<core::DeviceMemoryArena>(m, "DeviceMemoryArena")
        .def(py::init<size_t, size_t>(),
             py::arg("capacity_bytes"),
             py::arg("alignment") = core::DeviceMemoryArena::DEFAULT_ALIGNMENT,
             "Allocates fixed-capacity device memory arena with 256-byte alignment guarantee.")
        .def("capacity", &core::DeviceMemoryArena::capacity,
             "Returns total capacity in bytes.")
        .def("allocated_bytes", &core::DeviceMemoryArena::allocated_bytes,
             "Returns currently allocated bytes.")
        .def("remaining_bytes", &core::DeviceMemoryArena::remaining_bytes,
             "Returns remaining available bytes.")
        .def("alignment", &core::DeviceMemoryArena::alignment,
             "Returns sub-allocation byte alignment (e.g. 256).")
        .def("allocation_count", &core::DeviceMemoryArena::allocation_count,
             "Returns number of sub-allocations since last reset.")
        .def("reset", &core::DeviceMemoryArena::reset, py::arg("watermark") = 0,
             "Resets the bump allocator offset to 0 or specified watermark.");

    // Tensor Operations
    m.def("gemm", &py_gemm,
          py::arg("A"),
          py::arg("B"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          py::arg("backend") = "cuda_tiled",
          "Executes General Matrix Multiplication: C = alpha * (A @ B) + beta * C.\n"
          "Supported backends: 'cuda_tiled', 'cuda_naive', 'cublas', 'cpu'.");

    m.def("conv2d", &py_conv2d,
          py::arg("input"),
          py::arg("kernel"),
          py::arg("bias") = std::nullopt,
          py::arg("stride") = 1,
          py::arg("pad") = 0,
          py::arg("fuse_relu") = false,
          py::arg("backend") = "cuda",
          "Executes 2D Convolution forward pass for NCHW tensors with optional bias and fused ReLU.\n"
          "Supported backends: 'cuda', 'cpu'.");

    m.def("relu", &py_relu,
          py::arg("x"),
          py::arg("inplace") = false,
          py::arg("backend") = "cuda",
          "Executes elementwise ReLU activation: max(0, x).\n"
          "Supported backends: 'cuda', 'cpu'.");

    m.def("leaky_relu", &py_leaky_relu,
          py::arg("x"),
          py::arg("negative_slope") = 0.01f,
          py::arg("inplace") = false,
          py::arg("backend") = "cuda",
          "Executes elementwise Leaky ReLU activation: x if x > 0 else negative_slope * x.\n"
          "Supported backends: 'cuda', 'cpu'.");
}
