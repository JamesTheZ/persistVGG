extern "C" __global__ void default_function_kernel0( float* __restrict__ data,  float* __restrict__ kernel,  float* __restrict__ compute) {
	float compute_local[8];
	__shared__ float pad_temp_shared[1044];
	__shared__ float kernel_shared[216];
	float pad_temp_shared_local[12];
	float kernel_shared_local[6];
	for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
		compute_local[ff_c_init] = 0.000000e+00f;
		compute_local[(ff_c_init + 2)] = 0.000000e+00f;
		compute_local[(ff_c_init + 4)] = 0.000000e+00f;
		compute_local[(ff_c_init + 6)] = 0.000000e+00f;
	}
	for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
		if ((((int)threadIdx.z) * 261) < ((1044 - ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) - (((int)threadIdx.x) * 5))) {
			if ((((int)threadIdx.x) * 5) < (261 - ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)) {
				pad_temp_shared[(((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = ((((((1 - (((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 348) / 58)) <= (((int)blockIdx.y) * 4)) && ((((int)blockIdx.y) * 4) < (225 - (((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 348) / 58)))) && ((1 - ((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 58)) <= (((int)blockIdx.x) * 56))) && ((((int)blockIdx.x) * 56) < (225 - ((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 58)))) ? data[((((((((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 348) * 50176) + (((int)blockIdx.y) * 896)) + ((((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 348) / 58) * 224)) + (((int)blockIdx.x) * 56)) + ((((((int)threadIdx.z) * 261) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 58)) - 225)] : 0.000000e+00f);
			}
		}
	}
	if ((((int)threadIdx.z) * 2) < (8 - (((int)threadIdx.x) / 27))) {
		if ((((int)threadIdx.z) * 6) < (24 - (((int)threadIdx.x) / 9))) {
			if ((((int)threadIdx.z) * 18) < (72 - (((int)threadIdx.x) / 3))) {
				if ((((int)threadIdx.z) * 54) < (216 - ((int)threadIdx.x))) {
					if (((int)threadIdx.x) < 54) {
						kernel_shared[((((int)threadIdx.z) * 54) + ((int)threadIdx.x))] = kernel[(((((int)blockIdx.z) * 216) + (((int)threadIdx.z) * 54)) + ((int)threadIdx.x))];
					}
				}
			}
		}
	}
	__syncthreads();
	for (int rc_inner_outer = 0; rc_inner_outer < 3; ++rc_inner_outer) {
		for (int rx_inner_outer = 0; rx_inner_outer < 3; ++rx_inner_outer) {
			for (int ax2 = 0; ax2 < 3; ++ax2) {
				pad_temp_shared_local[ax2] = pad_temp_shared[((((rc_inner_outer * 348) + (ax2 * 58)) + ((int)threadIdx.x)) + rx_inner_outer)];
				pad_temp_shared_local[(ax2 + 3)] = pad_temp_shared[(((((rc_inner_outer * 348) + (ax2 * 58)) + ((int)threadIdx.x)) + rx_inner_outer) + 58)];
				pad_temp_shared_local[(ax2 + 6)] = pad_temp_shared[(((((rc_inner_outer * 348) + (ax2 * 58)) + ((int)threadIdx.x)) + rx_inner_outer) + 116)];
				pad_temp_shared_local[(ax2 + 9)] = pad_temp_shared[(((((rc_inner_outer * 348) + (ax2 * 58)) + ((int)threadIdx.x)) + rx_inner_outer) + 174)];
			}
			for (int ax0 = 0; ax0 < 2; ++ax0) {
				for (int ax21 = 0; ax21 < 3; ++ax21) {
					kernel_shared_local[((ax0 * 3) + ax21)] = kernel_shared[(((((((int)threadIdx.z) * 54) + (ax0 * 27)) + (rc_inner_outer * 9)) + (ax21 * 3)) + rx_inner_outer)];
				}
			}
			for (int ry_inner_inner = 0; ry_inner_inner < 3; ++ry_inner_inner) {
				for (int ff_c = 0; ff_c < 2; ++ff_c) {
					compute_local[ff_c] = (compute_local[ff_c] + (pad_temp_shared_local[ry_inner_inner] * kernel_shared_local[((ff_c * 3) + ry_inner_inner)]));
					compute_local[(ff_c + 2)] = (compute_local[(ff_c + 2)] + (pad_temp_shared_local[(ry_inner_inner + 3)] * kernel_shared_local[((ff_c * 3) + ry_inner_inner)]));
					compute_local[(ff_c + 4)] = (compute_local[(ff_c + 4)] + (pad_temp_shared_local[(ry_inner_inner + 6)] * kernel_shared_local[((ff_c * 3) + ry_inner_inner)]));
					compute_local[(ff_c + 6)] = (compute_local[(ff_c + 6)] + (pad_temp_shared_local[(ry_inner_inner + 9)] * kernel_shared_local[((ff_c * 3) + ry_inner_inner)]));
				}
			}
		}
	}
	for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
		compute[((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 100352)) + (ff_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 896)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x))] = compute_local[ff_inner_inner_inner];
		compute[(((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 100352)) + (ff_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 896)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 224)] = compute_local[(ff_inner_inner_inner + 2)];
		compute[(((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 100352)) + (ff_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 896)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 448)] = compute_local[(ff_inner_inner_inner + 4)];
		compute[(((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.z) * 100352)) + (ff_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 896)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 672)] = compute_local[(ff_inner_inner_inner + 6)];
	}
}


