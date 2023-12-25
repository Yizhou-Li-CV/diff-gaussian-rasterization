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

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	// R^TS^TSR to transform from a standard 3D gaussian
	// including, rotate first, scale on rotated axis, rotate back to world coord
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
// As viewmat and projmat is shared, this is for preparing 3D GS parameters for one camera only.
template<int C>
__global__ void preprocessCUDA(
	int P, // Input. the maximum number of 3D GS?
	int D, // Input. the degree (complexity?) of spherical harmonics
	int M, // Input. the max_coeffs for spherical harmonics
	const float* orig_points, // Input. 1-D array, every 3 element store a coordinate of point (center of 3D GS) 
	const glm::vec3* scales, // Input. 1-D array storing vec3, the scales of 3D GS for 3 axis
	const float scale_modifier, // Input. a constant to change the overall scale of 3D GS
	const glm::vec4* rotations, // Input. 1-D array storing vec4, the q4 vector expressing the rotation 
	const float* opacities, // Input. about 3D-to-2D GS projection (EWA)
	const float* shs, // Input. about spherical harmonics (will be converted to vec3* and get the 3 parameters of SH)
	bool* clamped, // Output. 1-D array storing RGB color. Store the bool of if RGB value is clamped (should be positive) or not for backward
	const float* cov3D_precomp, // Input. if 3D cov matrix is precomputed, can be used, otherwise (nullptr) would be recomputed
	const float* colors_precomp, // Input. similar. precomputed color
	const float* viewmatrix, // Input. w2c matrix (rotation + translation)
	const float* projmatrix, // Input. used to project point to camera's clip space (later converted to NDC space)
	const glm::vec3* cam_pos, // Input. 1-D array storting vec3. camera location
	const int W, int H, // Input. 
	const float tan_fovx, float tan_fovy, // Input. 
	const float focal_x, float focal_y, // Input. camera intrinsics for computing cov2D from cov3D
	int* radii, // Output. store radius of projected 2D gaussian (3xstd, ceil(3.f * sqrt(max(lambda1, lambda2))))
	float2* points_xy_image, // Output. 1-D array of 2floats. used to store projected center of 3D GS
	float* depths, // Output. 1-D array of 1 float. used to store the depth of 3D GS for current camera
	float* cov3Ds, // Output. 1-D array but store 6 numbers (3 vars and 3 covs) for 3D Gaussian covariance
	float* rgb, // Output. 1-D array to store 3 numbers (rgb with SH) of 3D GS to current camera
	float4* conic_opacity, // Output. About EWA
	const dim3 grid, // Input. Grid size. Suppose res is 160x160, block size is 16x16, so the grid.x=grid.y=10. since grid idx start from 0, the right bottom grid idx is (grid.x-1, grid.y-1).
	uint32_t* tiles_touched, // Output. 1-D array of the number of tiles that have been touched by 3D GS
	bool prefiltered // Input. if camera is filtered? because it is a single boolean.
	)
{
	// Get thread idx directly without manual calculation using cg lib
	// HERE the idx means the index of 3D GS, not the pixel!
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	// p_view is projected from orig_points using viewmatrix

	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// As done in in_frustum, here the p_proj means the NDC coordinates (using focal length + z + near)
	// for clip space, w is not divided yet. So here, it is NDC space, where [-1, 1] means inside camera frustum
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		// Cov3D (eq. 6) = RS(S^T)(R^T)
		// here con3D only store the upper right part of the matrix, including the variance of X, Y, Z and the covariance of XY, XZ, YZ
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		// update the pointer to the start of current cov3D
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// eq.5 in paper. Project 3D GS to 2D image coordinate with w2c matrix and focal length.
	// here cov is a 3-vector, in the order of var of X, CoV(X, Y), and var of Y
	// Here the p_orig is the center or mean of 3D GS
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 

	// average the variance of X and Z
	float mid = 0.5f * (cov.x + cov.z);
	// two eigenvalues of the CoV2D matrix (square of radius on each significant axis (the 2D gaussian ellipsoid can be seen as scaled first and stretched along the eigenvectors.)
	// so the sqrt of eigenvalues can be seen as the long axis radius and short axis radius
	// here find the long axis radius (long axis's std)
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// the gaussian itself actually have unlimited range
	// but to compute the tiles that have its influence, here simply define a circle with 3 * std of longer axis (for further range, the gaussian only have minor values that can be ignored) 
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// use only x and y to transform ndc coord to image coord.
	// the ndc coord actually already correspond to "camera coordinate", as it represents the frustum of the camera
	// so it should be easy to convert from ndc to image coord.
	// here the convert range from [-1, 1] becomes [-0.5, W-0.5] or [-0.5, H-0.5]
	// for example, for a image with 3x3 pixels, the general center is [2, 2] (center defined on right-bottom of center pixel)
	// however, here define it as [1.5, 1.5]. so the left-most coord is 1.5-2, right-most coord is 1.5 + 1 

	// here basically means the center of 3D gaussian that is projected onto the image coord
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// here grid defines the idx of blocks
	// for example, an image is 160x160, with block size 16x16, so the grid would be 10x10
	// get the minimal rectangle that can covers the whole circle
	// but actually return the tile index here (left top point's corresponding tile's x and y idx, and right bottom as well)
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// if this 3D GS is not influencing any pixels of the image plane (or screen space that is limited by H and W)
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		// here the idx means the index of 3D GS, not pixel
		// So here basically computes the color of current 3D GS based on the Spherical Harmonics
		// And each 3D GS has a unique color for different view direction
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	// Store the gaussian center on image coordinate for current gaussian with index idx 
	// (if this 3D GS is ignored and return before here, it would be nullptr)
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	// calculate the number of tiles that is been touched by current 3D GS
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
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
	const uint2* __restrict__ ranges, // Input. store 3D GS idx range for each block.
	const uint32_t* __restrict__ point_list, // Input. store [tile_idx | depth] for each 3D GS (queried by idx above)
	int W, int H, // Input
	const float2* __restrict__ points_xy_image, // Input. store 3D GS projected 2D coordinate (queried by [tile_idx | depth])
	const float* __restrict__ features,	// Input. features here store the color for each channel of this gaussian (pointed by collected_id[j])
	const float4* __restrict__ conic_opacity, // Input. About 3D to 2D GS projection
	float* __restrict__ final_T, // Output. final accumulated T for each pixel (thread)
	uint32_t* __restrict__ n_contrib, // Output. basically just stores the number of 3D GS influencing cur pixel that not ignored during splatting
	const float* __restrict__ bg_color, // Input. the RGB color of background. used when T is not small.
	float* __restrict__ out_color // Output. store the RGB color (so 3 numbers) of current pixel
	)
{
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
	// the global index of current pixel
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
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

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
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
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
			float4 con_o = collected_conic_opacity[j];
			// the influence level of the current 3D GS to current pixel, should be negative for opacity exp(-x) = 1 - alpha
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).

			// alpha (influence of current 3D gaussian to current pixel)
			float alpha = min(0.99f, con_o.w * exp(power));
			// if alpha is minor, we can ignore and continue to next GS
			if (alpha < 1.0f / 255.0f)
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
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}