#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "include/utility.hpp"

using namespace tt;

int main() {
  constexpr int device_id = 0;
  Device* device = CreateDevice(device_id);

  CommandQueue& cq = device->command_queue();
  Program program{};

  CoreRange mesh({0, 0}, {0, 0});

  constexpr auto num_tiles = 32;
  constexpr auto single_tile_size = 32 * 32 * sizeof(uint16_t);

  auto src0_cb_index = tt::CB::c_in0;
  tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
  auto buf_raw =
      tt::tt_metal::CreateBuffer({device, num_tiles * single_tile_size * 2,
                                  single_tile_size, BufferType::L1});
  CircularBufferConfig cb_src0_config =
      CircularBufferConfig(num_tiles * single_tile_size * 2,
                           {{src0_cb_index, cb_data_format}})
          .set_page_size(src0_cb_index, single_tile_size)
          .set_globally_allocated_address(*buf_raw);
  auto cb_src0 =
      tt::tt_metal::CreateCircularBuffer(program, mesh, cb_src0_config);
  auto cb_src0_addr = buf_raw->address();
  auto reader_id = tt_metal::CreateKernel(
      program,
      get_absolute_path("kernels/dataflow/"
                        "summa_read.cpp"),
      mesh,
      tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default,
                                   .compile_args = {}});

  auto writer_id = tt_metal::CreateKernel(
      program,
      get_absolute_path("kernels/dataflow/"
                        "summa_write.cpp"),
      mesh,
      tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                   .noc = NOC::RISCV_0_default,
                                   .compile_args = {}});

  auto compute_kernels = tt_metal::CreateKernel(
      program,
      get_absolute_path("kernels/"
                        "compute/summa.cpp"),
      mesh,
      tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                              .compile_args = {}});

  EnqueueProgram(cq, program, false);

  CloseDevice(device);
}
