/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template<int C>
__global__ void preprocessCUDA(
	int P, // Input. the maximum number of kernels
	const float2* orig_points, // Input. 1-D array, store a 2D coordinate of point (center of kernel) 
	const int W, int H, // Input. 
	const int* radii, // Input. store radius of 2D kernels (CoC)
	const dim3 grid, // Input. Grid size. Suppose res is 160x160, block size is 16x16, so the grid.x=grid.y=10. since grid idx start from 0, the right bottom grid idx is (grid.x-1, grid.y-1).
	uint32_t* tiles_touched, // Output. 1-D array of the number of tiles that have been touched by 3D GS
)
{
	// Get thread idx directly without manual calculation using cg lib
	// HERE the idx means the index of 3D GS, not the pixel!
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	tiles_touched[idx] = 0;

	// here basically means the center of 3D gaussian that is projected onto the image coord
	float2 point_image = orig_points[idx];
	uint2 rect_min, rect_max;
	// here grid defines the idx of blocks
	// for example, an image is 160x160, with block size 16x16, so the grid would be 10x10
	// get the minimal rectangle that can covers the whole circle
	// but actually return the tile index here (left top point's corresponding tile's x and y idx, and right bottom as well)
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// if this 3D GS is not influencing any pixels of the image plane (or screen space that is limited by H and W)
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

void FORWARD::preprocess(int P,
	const float2* orig_points,
	const int W, int H,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched
	)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		orig_points,
		W,
		H,
		radii,
		grid,
		tiles_touched
	);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// CHANNELS is a uint32_t number, so this function can be compiled to different channel numbers
// Not use as input parameter here, because compiler will know the channel number in advance for better optimization
template <uint32_t CHANNELS>
// tell compiler the block size for optimization
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges, // Input. store 3D GS range (after duplication and sorting) for each block. Done in identifyTileRanges.
	const uint32_t* __restrict__ point_list, // Input. store actual 3D GS index (actual idx before duplication). query the true 3D GS index for current tile with range.
	int W, int H, // Input
	const float2* __restrict__ points_xy_image, // Input. store 3D GS projected 2D coordinate (queried by [tile_idx | depth])
	const float* __restrict__ features,	// Input. features here store the color for each channel of this gaussian (pointed by collected_id[j])
	float* __restrict__ final_T, // Output. final accumulated T for each pixel (thread)
	uint32_t* __restrict__ n_contrib, // Output. basically just stores the number of 3D GS influencing cur pixel that not ignored during splatting
	const float* __restrict__ bg_color, // Input. the RGB color of background. used when T is not small.
	float* __restrict__ out_color // Output. store the RGB color (so 3 numbers) of current pixel
	const int* radii // Input. the radius of each kernel (should be replicated during sorting)
){
	// Identify current tile and associated min/max pixel range.
	// get block index for current thread (pixel)
	auto block = cg::this_thread_block();
	// horizontal block numbers
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// the global pixel coordinates of current blocks' left top pixel
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	// the global pixel coordinates of current blocks' right bottom pixel
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// the global pixel coordinates of current pixel in current block
	// thread_index() return a 2D thread index (y, x)
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// the global index of current pixel. here the pix.x and pix.y means global coord of current thread
	// so the global index can be calculated
	uint32_t pix_id = W * pix.y + pix.x;
	// into float global coordinate for current pixel
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// group_index() here return the 2D block index (y, x)

	// notice here the 3DGS are sorted by key of [tile_idx | 3D GS depth]
	// so for current block (tile), we can easily find the ranges of 3DGS indexes that need to be considered within this block (tile) 

	// find the range for current block 
	// (blocks are placed from left to right, from top to bottom, in a flatten style)
	// so needs to calculate the block index here
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// y and x here means the left bound and right bound
	// BLOCK size here is 16*16. For each iteration, it iterates 16*16 times (a batch is 16*16)
	// here is to limit the number to be processed in parallel
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// left number of index to be processed
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// the data here is shared by the whole block with fast access
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	// __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	// accumulated transmittance for current pixel
	float T = 1.0f;

	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	// accumulated color for current pixel
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	// here each thread (pixel) will go over this iteration. this iteration is to iterate over 3D gaussians (3D gaussians are divided into batches)
	// each pixel should iterate over all 3D GS for this tile. But not using a single loop, but a two-layer loop.
	// remember the cuda process here is written for one thread
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// __syncthreads_count here stop every threads in a block here
		// and count how many reached threads (pixels) shows "done"

		// if not all done for current batch, still require all threads to help load data into the shared memory
		// (even though done threads won't go into next iteration)
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// thread_rank() returns the flat index of current thread inside the block
		// though one pixel needs to iterate over all 3D GS, but for shared memory usage, this is not necessary
		// batch split is the same for all threads. each thread only take care of loading the "point list"
		// here is why the batch is 16*16: each thread loads one point into the shared memory, with 16*16 threads, each time 16*16 points can be loaded in parallel into the shared memory
		// also, shared memory has limited size. for current batch, only load the data for current batch
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			// load a batch (16*16) of points (3D GS index and center coords) into the shared memory
			// and progress make it possible for all thread to each item of data in the batch into the shared memory in parallel
			int coll_id = point_list[range.x + progress];
			// the 3D GS index
			collected_id[block.thread_rank()] = coll_id;
			// the 3D GS center coord in image coordinate
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			// collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		// wait until all threads finish loading
		block.sync();

		// Iterate over all 3D gaussians in this batch
		// because data about a batch of 3D GS is loaded into the shared memory, the query would be very fast
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			// x_dist and y_dist from current pixel to current 3D GS center
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float dist_sq = d.x * d.x + d.y * d.y
			float rad = (float)radii[j]; // radius of current kernel
			// float4 con_o = collected_conic_opacity[j];
			// the influence level of the current 3D GS to current pixel, should be negative for opacity exp(-x) = 1 - alpha
			// float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			// if (power > 0.0f)
			// 	continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).

			// alpha (influence of current 3D gaussian to current pixel)
			// if too far from kernel center, alpha is zero
			if(dist_sq > rad * rad)
				float alpha = 0f;
			// else, the alpha here is simply calculated as this
			// but for diff of depth, we need to introduce depth here in the alpha calculate
			else
				float alpha = 1f / (3.1415926f * rad * rad);
			// if alpha is minor, we can ignore and continue to next GS
			if (alpha < 1.0f / (255.0f * 10))
				continue;
			// T = T * (1 - alpha)
			float test_T = T * (1 - alpha);
			// early stop if accumulated transmittance is too small (already high opacity from front to current layer)
			if (test_T < 0.0001f)
			{
				// continue but quit the iteration here
				// later batch also not needed to be considered
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				// color += c * alpha * T (similar to NeRF, just alpha blending from front to back)
				// features here store the color for each channel of this gaussian (pointed by collected_id[j])
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.

			// basically just stores the number of 3D GS influencing cur pixel that not ignored during splatting
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// the accumulated transmittance for current pixel (can be used to scale the splatted results?)
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			// add to global buffer for final output. Here the left T will be used to add a bg_color (possibly the sky color)
			// or for dof splatting, we can use T to scale the final color to make the rendering results look more natural
			out_color[ch * H * W + pix_id] = C[ch] / T;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ colors,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color
	const int* radii // Input. the radius of each kernel (should be replicated during sorting)
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		points_xy_image,
		colors,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		radii);
}