#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/holder_pool_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void GroupHolderPoolForward(const int n, const int channels, const int dim,
    const bool exp_p, const Dtype max_p, const Dtype min_p, const Dtype* in, 
    Dtype* out, Dtype* p_data, const Dtype d, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype p = p_data[c];
    if (exp_p){
      p = exp(p);
    }
    if (p > max_p){
      p = max_p;
      p_data[c] = max_p;
      if (exp_p){
        p_data[c] = log(max_p);
      }
    }
    if (p < min_p){
      p = min_p;
      p_data[c] = min_p;
      if (exp_p){
        p_data[c] = log(min_p);
      }
    }
    Dtype q = 1./(p-1.);
    out[index] = -d* pow((abs(in[index])+Dtype(1e-10)),q) * in[index] 
                    / pow((pow(in[index],Dtype(2))+Dtype(1e-20)),Dtype(0.5));
  }
}

template <typename Dtype>
void HolderPoolLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc) {
    channels = 1;
    dim = bottom[0]->channels();
  }
  if (!fc && channel_shared_){
    //LOG(INFO) << "shared_pool!";
    const int div_factor = channel_shared_ ? channels/group : 1;
    Dtype* p_data = this->blobs_[0]->mutable_cpu_data();

    /*Dtype* f_data_1 = forward_buff_1.mutable_gpu_data();
    Dtype* f_diff_1 = forward_buff_1.mutable_gpu_diff();
    Dtype* f_data_2 = forward_buff_2.mutable_gpu_data();
    caffe_gpu_abs(count, bottom_data, f_data_1);
    caffe_gpu_powx(count, f_data_1, q, f_diff_1);
    caffe_gpu_powx(count, bottom_data, Dtype(2), f_data_1);
    caffe_gpu_add_scalar(count, Dtype(1e-20), f_data_1);
    caffe_gpu_powx(count, f_data_1, Dtype(0.5), f_data_2);
    caffe_gpu_div(count, bottom_data, f_data_2, f_data_1);
    caffe_gpu_mul(count, f_data_1, f_diff_1, f_data_2);
    caffe_gpu_scale(count, -d, f_data_2, top_data);*/

    // NOLINT_NEXT_LINE(whitespace/operators)
    GroupHolderPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, exp_p, max_p, min_p, bottom_data, top_data, p_data, d, div_factor);
    CUDA_POST_KERNEL_CHECK;
  }else{
    Dtype* p_data = this->blobs_[0]->mutable_cpu_data();
    const int div_factor = channel_shared_ ? channels/group : 1;
    Dtype* f_data_1 = forward_buff_1.mutable_gpu_data();
    Dtype* f_diff_1 = forward_buff_1.mutable_gpu_diff();
    Dtype* f_data_2 = forward_buff_2.mutable_gpu_data();
    for (int i = 0; i < num*channels; ++i) {
      int c = i % channels / div_factor;
      Dtype p = p_data[c];
      if (exp_p){
        p = exp(p);
      }
      if (p > max_p){
        p = max_p;
        if (exp_p){
          p_data[c] = log(max_p);
        }
      }
      if (p < min_p){
        p = min_p;
        if (exp_p){
          p_data[c] = log(min_p);
        }
      }
      Dtype q = 1. / (p - 1.);
      caffe_gpu_abs(dim, bottom_data, f_data_1);
      caffe_gpu_powx(dim, f_data_1, q, f_diff_1);
      caffe_gpu_powx(dim, bottom_data, Dtype(2), f_data_1);
      caffe_gpu_add_scalar(dim, Dtype(1e-20), f_data_1);
      caffe_gpu_powx(dim, f_data_1, Dtype(0.5), f_data_2);
      caffe_gpu_div(dim, bottom_data, f_data_2, f_data_1);
      caffe_gpu_mul(dim, f_data_1, f_diff_1, f_data_2);
      caffe_gpu_scale(dim, -d, f_data_2, top_data);
      bottom_data += dim;
      top_data += dim;
    }
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  //HolderPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //    count, channels, dim, bottom_data, top_data, p_data, d, div_factor);
  //CUDA_POST_KERNEL_CHECK;
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void HolderPoolBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* p_data, const Dtype d, const int div_factor, const bool exp_p) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype p = p_data[c];
    if (exp_p){
      p = exp(p);
    }
    Dtype q = 1. / (p - 1.);
    Dtype denom = (pow(in_data[index],Dtype(2))+p*Dtype(1e-20)) / pow((pow(in_data[index],Dtype(2)) + Dtype(1e-20)),Dtype(1.5));
    out_diff[index] = -d * q * in_diff[index] * pow((abs(in_data[index])+Dtype(1e-10)),q) * denom;
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void HolderPoolParamBackward(const int n, const int rows, 
    const int rowPitch, const int dim, const int channels, const Dtype* p_data, 
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff, 
    const Dtype d, const bool exp_p,const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype p = p_data[c];
    Dtype exp_diff = 1;
    if (exp_p){
      p = exp(p);
      exp_diff = p;
    }
    Dtype q = 1. / (p_data[c]-1.);
    Dtype coef = -1./pow((p_data[c]-1.),2);
    Dtype sgn = in_data[index] / sqrt(pow(in_data[index],2)+Dtype(1e-20));
    out_diff[index] = -d * sgn * coef * in_diff[index] * pow((abs(in_data[index])+Dtype(1e-10)),q)
                       * log(sqrt(pow(in_data[index],2)+Dtype(1e-20))) * exp_diff;
    for ( int k = 1; k < rows; k++ ) {
      sgn = in_data[index + k*rowPitch] / sqrt(pow(in_data[index + k*rowPitch],2)+Dtype(1e-20));
      out_diff[index] += -d *sgn * coef * in_diff[index + k*rowPitch]
           * pow((abs(in_data[index+k*rowPitch])+Dtype(1e-10)),q)
           * log(sqrt(pow(in_data[index+k*rowPitch],2)+Dtype(1e-20))) * exp_diff;
    }
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void HolderPoolParamBackward1(const int n, const int dim, 
    const int channels, const Dtype* p_data, 
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff, 
    const Dtype d, const bool exp_p,const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    Dtype p = p_data[c];
    Dtype exp_diff = 1;
    if (exp_p){
      p = exp(p);
      exp_diff = p;
    }
    Dtype q = 1. / (p_data[c]-1.);
    Dtype coef = -1./pow((p_data[c]-1.),2);
    Dtype sgn = in_data[index] / sqrt(pow(in_data[index],2)+Dtype(1e-10));
    out_diff[index] = -d * sgn * coef * in_diff[index] * pow((abs(in_data[index])+Dtype(1e-10)),q)
                       * log(sqrt(pow(in_data[index],2)+Dtype(1e-10))) * exp_diff;
  }
}

template <typename Dtype>
void HolderPoolLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* p_data = this->blobs_[0]->gpu_data();
  const int count = bottom[0]->count();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }
  const int div_factor = channel_shared_ ? channels/group : 1;
  // Propagate to param
  if (this->param_propagate_down_[0]) {
    Dtype* p_diff = this->blobs_[0]->mutable_gpu_diff();
    //Dtype* f_diff_1 = forward_buff_1.mutable_gpu_diff();
    //int cdim = channels * dim;
    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    HolderPoolParamBackward1<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(
      count, dim, channels, p_data, 
      top_diff, bottom_data, forward_buff_1.mutable_gpu_diff(), d, exp_p,div_factor);
    CUDA_POST_KERNEL_CHECK;
    /*HolderPoolParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), dim, channels, p_data, 
      top_diff, bottom_data, forward_buff_1.mutable_gpu_diff(), d, exp_p,div_factor);
    CUDA_POST_KERNEL_CHECK;*/
    if (channel_shared_) {
      if (group==1){
        Dtype dsum;
        caffe_gpu_dot<Dtype>(count, forward_buff_1.gpu_diff(),
           multiplier_.gpu_data(), &dsum);
        caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), p_diff);
      }else{
        caffe_gpu_gemv<Dtype>(CblasNoTrans, group, channels/group*dim, 1.,
          forward_buff_1.gpu_diff(), multiplier_.gpu_data(), 1.,
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
    HolderPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, p_data,
        d, div_factor, exp_p);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(HolderPoolLayer);


}  // namespace caffe
