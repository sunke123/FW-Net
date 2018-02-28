#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/sig_scale_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void SigScaleForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* p_data, const bool counterpart, 
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype r = 1. / (1. + exp(-p_data[c]));
    if (counterpart){
      r = 1. - 1. / (1. + exp(-p_data[c]));
    }
    out[index] = in[index]*r;
  }
}

template <typename Dtype>
void SigScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }
  const Dtype* p_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels/group : 1;

  // NOLINT_NEXT_LINE(whitespace/operators)
  SigScaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, p_data, counterpart, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void SigScaleBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, Dtype* out_diff, const bool counterpart, 
    const Dtype* p_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype r = 1. / (1. + exp(-p_data[c]));
    if (counterpart){
      r = 1. - 1. / (1. + exp(-p_data[c]));
    }
    out_diff[index] = in_diff[index] * r;
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void SigScaleParamBackward(const int n,
    const int rows, const int dim, const int channels,
    const Dtype* p_data, const bool counterpart, const int rowPitch,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels;
    Dtype r = exp(-p_data[c]) / pow((1+exp(-p_data[c])),2);
    if (counterpart){
      r = -r;
    }
    out_diff[index] = in_diff[index] * in_data[index] * r;
    for ( int k = 1; k < rows; k++ ) {
      out_diff[index] += in_diff[index + k*rowPitch]
          * in_data[index + k*rowPitch] * r;
    }
  }
}

template <typename Dtype>
void SigScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }

  // Propagate to param
  if (this->param_propagate_down_[0]) {
    const Dtype* p_data = this->blobs_[0]->gpu_data();
    Dtype* p_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;
    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigScaleParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), dim, channels, p_data, counterpart, 
      top[0]->offset(1), top_diff, bottom_data,
      backward_buff_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      if (group == 1){
        Dtype dsum;
        caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
          multiplier_.gpu_data(), &dsum);
        //LOG(INFO) << "p_diff: "<<dsum;
        caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), p_diff);
      }else{
        caffe_gpu_gemv<Dtype>(CblasNoTrans, group, channels/group*dim, 1.,
          backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
          p_diff);        
      }
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        p_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* p_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels/group : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigScaleBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_diff, 
        counterpart, p_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SigScaleLayer);


}  // namespace caffe
