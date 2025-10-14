/*
 * This file is part of Vlasiator.
 * Copyright 2010-2024 Finnish Meteorological Institute and University of Helsinki
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef GPU_BASE_H
#define GPU_BASE_H

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "arch_device_api.h"

// Extra profiling stream synchronizations?
#define SSYNC CHK_ERR( gpuStreamSynchronize(stream) )
//#define SSYNC

#include <stdio.h>
#include "include/splitvector/splitvec.h"
#include "include/hashinator/hashinator.h"
#include "../definitions.h"
#include "../vlasovsolver/vec.h"
#include "../velocity_mesh_parameters.h"
#include <phiprof.hpp>

static const double BLOCK_ALLOCATION_PADDING = 1.5;
static const double BLOCK_ALLOCATION_FACTOR = 1.2;
// buffers need to be larger for translation
static const int TRANSLATION_BUFFER_ALLOCATION_FACTOR = 5;

#define DIMS 1
// Maximum number of CPU threads supported for per-thread GPU resources.
// Increased to avoid out-of-bounds when OMP uses >64 threads on some clusters.
#define MAXCPUTHREADS 256

void gpu_init_device();
void gpu_clear_device();
gpuStream_t gpu_getStream();
gpuStream_t gpu_getPriorityStream();
uint gpu_getThread();
uint gpu_getMaxThreads();
int gpu_getDevice();
uint gpu_getAllocatedThreads();

// Optional low-level GPU pointer debugging helpers (CUDA/HIP specific)
// Enable verbose pointer attribute prints by defining GPU_DEBUG_PTRS in compile flags
#if defined(USE_GPU)
#if defined(__CUDACC__)
#include <cuda_runtime.h>
inline static const char* gpu_mem_type_str(cudaMemoryType t) {
   switch (t) {
      case cudaMemoryTypeUnregistered: return "unregistered";
      case cudaMemoryTypeHost:         return "host";
      case cudaMemoryTypeDevice:       return "device";
      case cudaMemoryTypeManaged:      return "managed";
      default:                         return "unknown";
   }
}
inline static void gpu_debug_pointer(const void* p, const char* name) {
#ifdef GPU_DEBUG_PTRS
   cudaPointerAttributes attr;
   cudaError_t st = cudaPointerGetAttributes(&attr, p);
   if (st == cudaSuccess) {
#if CUDART_VERSION >= 10000
      // Newer CUDA uses attr.type
      printf("[GPU-DBG] %s=%p type=%s device=%d devicePointer=%p hostPointer=%p\n",
             name, p,
             (attr.type==cudaMemoryTypeDevice?"device":(attr.type==cudaMemoryTypeHost?"host":(attr.type==cudaMemoryTypeManaged?"managed":"unknown"))),
             attr.device,
             attr.devicePoin  ter,
             attr.hostPointer);
#else
      // Older CUDA uses attr.memoryType
      printf("[GPU-DBG] %s=%p type=%s devicePointer=%p hostPointer=%p\n",
             name, p, gpu_mem_type_str(attr.memoryType), attr.devicePointer, attr.hostPointer);
#endif
   } else {
      printf("[GPU-DBG] %s=%p cudaPointerGetAttributes error: %s\n", name, p, cudaGetErrorString(st));
   }
#else
   (void)p; (void)name; // no-op
#endif
}
#if defined(__CUDACC__)
inline static bool gpu_can_memop(const void* p) {
   if (!p) return false;
   cudaPointerAttributes attr;
   if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) return false;
#if CUDART_VERSION >= 10000
   return (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
#else
   return (attr.memoryType == cudaMemoryTypeDevice || attr.memoryType == cudaMemoryTypeManaged);
#endif
}
#else
inline static bool gpu_can_memop(const void* p) { return p != nullptr; }
#endif

// Safe, logging memset-async for debug; returns true if memset attempted and succeeded
inline static bool gpu_try_memset_async(void* ptr, int value, size_t bytes, gpuStream_t stream) {
   if (!gpu_can_memop(ptr)) return false;
#if defined(__CUDACC__)
   cudaError_t err = cudaMemsetAsync(ptr, value, bytes, stream);
   if (err != cudaSuccess) {
      printf("[GPU-DBG] cudaMemsetAsync failed on %p: %s (skipping)\n", ptr, cudaGetErrorString(err));
      return false;
   }
   return true;
#elif defined(__HIP_PLATFORM_HCC___)
   hipError_t err = hipMemsetAsync(ptr, value, bytes, stream);
   if (err != hipSuccess) {
      printf("[GPU-DBG] hipMemsetAsync failed on %p: %d (skipping)\n", ptr, (int)err);
      return false;
   }
   return true;
#else
   (void)ptr; (void)value; (void)bytes; (void)stream; return false;
#endif
}
#elif defined(__HIP_PLATFORM_HCC___)
#include <hip/hip_runtime.h>
inline static void gpu_debug_pointer(const void* p, const char* name) {
#ifdef GPU_DEBUG_PTRS
   hipPointerAttribute_t attr{};
   hipError_t st = hipPointerGetAttributes(&attr, const_cast<void*>(p));
   if (st == hipSuccess) {
      printf("[GPU-DBG] %s=%p memoryType=%d device=%d devicePointer=%p hostPointer=%p\n",
             name, p, attr.memoryType, attr.device, attr.devicePointer, attr.hostPointer);
   } else {
      printf("[GPU-DBG] %s=%p hipPointerGetAttributes error: %d\n", name, p, (int)st);
   }
#else
   (void)p; (void)name;
#endif
}
#else
inline static void gpu_debug_pointer(const void* p, const char* name) { (void)p; (void)name; }
#endif
#else
inline static void gpu_debug_pointer(const void* p, const char* name) { (void)p; (void)name; }
#endif

void gpu_vlasov_allocate(uint maxBlockCount);
void gpu_vlasov_deallocate();
void gpu_vlasov_allocate_perthread(uint cpuThreadID, uint blockAllocationCount);
void gpu_vlasov_deallocate_perthread(uint cpuThreadID);
uint gpu_vlasov_getAllocation();
uint gpu_vlasov_getSmallestAllocation();

void gpu_acc_allocate(uint maxBlockCount);
void gpu_acc_allocate_perthread(uint cpuThreadID, uint columnAllocationCount);
void gpu_acc_deallocate();
void gpu_acc_deallocate_perthread(uint cpuThreadID);

void gpu_blockadjust_allocate(uint maxBlockCount);
void gpu_blockadjust_allocate_perthread(uint cpuThreadID, uint maxBlockCount);
void gpu_blockadjust_deallocate();
void gpu_blockadjust_deallocate_perthread(uint cpuThreadID);
   
void gpu_trans_allocate(cuint nAllCells=0,
                        cuint sumOfLengths=0,
                        cuint largestVmesh=0,
                        cuint unionSetSize=0,
                        cuint transGpuBlocks=0,
                        cuint nPencils=0);
void gpu_trans_deallocate();

extern gpuStream_t gpuStreamList[];
extern gpuStream_t gpuPriorityStreamList[];

// Unified memory class for inheritance
class Managed {
public:
   void *operator new(size_t len) {
      void *ptr;
      CHK_ERR(gpuMallocManaged(&ptr, len));
      CHK_ERR(gpuDeviceSynchronize());
      return ptr;
   }

   void operator delete(void *ptr) {
      CHK_ERR(gpuDeviceSynchronize());
      CHK_ERR(gpuFree(ptr));
   }

   void* operator new[] (size_t len) {
      void *ptr;
      CHK_ERR(gpuMallocManaged(&ptr, len));
      CHK_ERR(gpuDeviceSynchronize());
      return ptr;
   }

   void operator delete[] (void* ptr) {
      CHK_ERR(gpuDeviceSynchronize());
      CHK_ERR(gpuFree(ptr));
   }

};

// Structs used by Vlasov Acceleration semi-Lagrangian solver
struct Column {
   int valuesOffset;                              // Source data values
   size_t targetBlockOffsets[MAX_BLOCKS_PER_DIM]; // Target data array offsets
   int nblocks;                                   // Number of blocks in this column
   int minBlockK,maxBlockK;                       // Column parallel coordinate limits
   int kBegin;                                    // Actual un-sheared starting block index
   int i,j;                                       // Blocks' perpendicular coordinates
};

struct ColumnOffsets {
   split::SplitVector<uint> columnBlockOffsets; // indexes where columns start (in blocks, length totalColumns)
   split::SplitVector<uint> columnNumBlocks; // length of column (in blocks, length totalColumns)
   split::SplitVector<uint> setColumnOffsets; // index from columnBlockOffsets where new set of columns starts (length nColumnSets)
   split::SplitVector<uint> setNumColumns; // how many columns in set of columns (length nColumnSets)

   ColumnOffsets(uint nColumns) {
      columnBlockOffsets.resize(nColumns);
      columnNumBlocks.resize(nColumns);
      setColumnOffsets.resize(nColumns);
      setNumColumns.resize(nColumns);
      columnBlockOffsets.clear();
      columnNumBlocks.clear();
      setColumnOffsets.clear();
      setNumColumns.clear();
      // These vectors themselves are not in unified memory, just their content data
      gpuStream_t stream = gpu_getStream();
      columnBlockOffsets.optimizeGPU(stream);
      columnNumBlocks.optimizeGPU(stream);
      setColumnOffsets.optimizeGPU(stream);
      setNumColumns.optimizeGPU(stream);
   }
   void prefetchDevice(gpuStream_t stream) {
      columnBlockOffsets.optimizeGPU(stream);
      columnNumBlocks.optimizeGPU(stream);
      setColumnOffsets.optimizeGPU(stream);
      setNumColumns.optimizeGPU(stream);
   }
};

// Device data variables, to be allocated in good time. Made into an array so that each thread has their own pointer.
extern vmesh::LocalID *gpu_GIDlist[];
extern vmesh::LocalID *gpu_LIDlist[];
extern vmesh::GlobalID *gpu_BlocksID_mapped[];
extern vmesh::GlobalID *gpu_BlocksID_mapped_sorted[];
extern vmesh::GlobalID *gpu_LIDlist_unsorted[];
extern vmesh::LocalID *gpu_columnNBlocks[];

extern Vec *gpu_blockDataOrdered[];
extern uint *gpu_cell_indices_to_id[];
extern uint *gpu_block_indices_to_id[];
extern uint *gpu_vcell_transpose;

extern Vec** host_pencilOrderedPointers;
extern Vec** dev_pencilOrderedPointers;
extern Realf** dev_pencilBlockData;
extern uint* dev_pencilBlocksCount;

extern void *gpu_RadixSortTemp[];
extern uint gpu_acc_RadixSortTempSize[];

extern Real *returnReal[];
extern Realf *returnRealf[];
extern vmesh::LocalID *returnLID[];
extern vmesh::GlobalID *invalidGIDpointer;

extern Column *gpu_columns[];
extern ColumnOffsets *cpu_columnOffsetData[];
extern ColumnOffsets *gpu_columnOffsetData[];

// Hash map and splitvectors used in block adjustment
extern split::SplitVector<vmesh::GlobalID> *gpu_list_with_replace_new[];
extern split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *gpu_list_delete[];
extern split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *gpu_list_to_replace[];
extern split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *gpu_list_with_replace_old[];
// Device copies of SplitVector objects for safe kernel use
extern split::SplitVector<vmesh::GlobalID> *gpu_list_with_replace_new_dev[];
extern split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *gpu_list_delete_dev[];
extern split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *gpu_list_to_replace_dev[];
extern split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>> *gpu_list_with_replace_old_dev[];

// SplitVector information structs for use in fetching sizes and capacities without page faulting
// extern split::SplitInfo *info_1[];
// extern split::SplitInfo *info_2[];
// extern split::SplitInfo *info_3[];
// extern split::SplitInfo *info_4[];
// extern Hashinator::MapInfo *info_m[];

// Vectors and set for use in translation, actually declared in vlasovsolver/gpu_trans_map_amr.hpp
// to sidestep compilation errors
// extern split::SplitVector<vmesh::VelocityMesh*> *allVmeshPointer;
// extern split::SplitVector<vmesh::VelocityMesh*> *allPencilsMeshes;
// extern split::SplitVector<vmesh::VelocityBlockContainer*> *allPencilsContainers;
// extern split::SplitVector<vmesh::GlobalID> *unionOfBlocks;
// extern Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *unionOfBlocksSet;

// Counters used in allocations
extern uint gpu_vlasov_allocatedSize[];
extern uint gpu_blockadjust_allocatedSize[];
extern uint gpu_acc_allocatedColumns;
extern uint gpu_acc_columnContainerSize;
extern uint gpu_acc_foundColumnsCount;

#endif
