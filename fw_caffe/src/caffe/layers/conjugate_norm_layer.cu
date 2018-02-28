#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conjugate_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
Dtype FindMax(const int count, const Dtype* inp) {
  Dtype maxinp = abs(inp[0]);
  for (int i=1; i<count; i++){
    if (abs(inp[i]) > maxinp){
      maxinp = abs(inp[i]);
    }
  }
  return maxinp;
}

template <typename Dtype>
void PNorm(const int count, const Dtype p, const Dtype* inp, 
    Dtype* f_buff, Dtype &n_buff, Dtype* out) {
  caffe_gpu_abs(count, inp, f_buff);
  //caffe_gpu_add_scalar(dim, Dtype(1e-10), b_buff_data);
  caffe_gpu_powx(count, f_buff, p, out);
  Dtype denom;
  caffe_gpu_asum(count, out, &denom);
  denom = pow(denom, 1./p);
  //LOG(INFO) << "denom: " << denom;
  n_buff = denom;
  caffe_gpu_scale(count, Dtype(1./denom), inp, out);
}

template <typename Dtype>
void ConjugateNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }
  //LOG(INFO) << "c: " << channels << " dim: "<<dim;
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  Dtype* n_buff = norm_buff_.mutable_cpu_data();
  Dtype* f_buff = forward_buff_.mutable_gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < num*channels; ++i) {
    int c = i % channels / div_factor;
    if (p_data[c] < 1.1){
      n_buff[i] = FindMax(dim, bottom[0]->cpu_data()+i*dim);
      caffe_gpu_scale(dim, Dtype(1./n_buff[i]), bottom_data, top_data);
    }else{
      Dtype q = p_data[c] / (p_data[c] - 1.);
      PNorm(dim, q, bottom_data, f_buff, n_buff[i], top_data);
    }
    bottom_data += dim;
    top_data += dim;
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void ConjugateNormBackward2(const int n, const Dtype pnorm, 
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if(abs(in_data[index]) == pnorm){
      out_diff[index] = 0.0;
    }else{
      out_diff[index] = in_diff[index] / pnorm;
    }
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void ConjugateNormBackward(const int n, const Dtype p,
    const Dtype pnorm, const Dtype inpxdiff, const Dtype* in_diff, 
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] / pnorm - inpxdiff * in_data[index]
                      * pow((abs(in_data[index])+Dtype(1e-10)),(p-2)) / pow(pnorm, (p+1));
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void ConjugateNormParamBackward(const int n, 
    const Dtype pnorm, const Dtype m, 
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] += in_diff[index] * in_data[index] / pnorm * m;
  }
}

template <typename Dtype>
void ConjugateNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  const Dtype* n_buff = norm_buff_.cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }
  int div_factor = channel_shared_ ? channels : 1;

  // Propagate to param
  if (this->param_propagate_down_[0]) {
    Dtype* p_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* b_buff_data = backward_buff_.mutable_gpu_data();
    Dtype* b_buff_diff = backward_buff_.mutable_gpu_diff();
    Dtype* b_diff_2 = backward_buff_2.mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    int cdim = channels * dim;
    caffe_gpu_set(cdim, Dtype(0.), b_diff_2);

    for (int i=0; i<num*channels; i++){
      int c = i % channels / div_factor;
      Dtype* b_buff_diff_2 = b_diff_2 + (i%channels) *dim;
      if (p_data[c] < 1.1){
        caffe_gpu_set(dim,Dtype(0),b_buff_diff_2);
      }else{
        Dtype p = p_data[c];
        Dtype q = p / (p-1.);
        Dtype qnorm = n_buff[i];
        Dtype s1 = -(1./p/(p-1.)) * log(qnorm);
        Dtype s2 = (1./p/(p-1.)) * 1./pow(qnorm,q);
        caffe_copy(dim, bottom_data+i*dim, b_buff_data);
        caffe_gpu_add_scalar(dim, Dtype(1e-20), b_buff_data);
        caffe_gpu_powx(dim, b_buff_data, Dtype(2), b_buff_diff);
        caffe_gpu_powx(dim, b_buff_diff, Dtype(0.5), b_buff_data);
        caffe_gpu_log(dim, b_buff_data, bottom_diff);
        caffe_gpu_powx(dim, b_buff_data, q, b_buff_diff);
        Dtype s3;
        caffe_gpu_dot(dim, bottom[0]->gpu_diff(), backward_buff_.gpu_diff(), &s3);
        s3 = s1 + s2*s3;
        //caffe_gpu_set(dim, Dtype(0.), b_buff_diff);
        // NOLINT_NEXT_LINE(whitespace/operators)
        ConjugateNormParamBackward<Dtype><<<CAFFE_GET_BLOCKS(dim),
          CAFFE_CUDA_NUM_THREADS>>>(
          dim, qnorm, s3, top_diff+i*dim, bottom_data+i*dim, b_buff_diff_2);
        CUDA_POST_KERNEL_CHECK;
      }
    }
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_2.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), p_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        p_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    for(int i=0; i<num*channels; i++){
      int c = i % channels / div_factor;
      if (p_data[c] < 1.1){
        ConjugateNormBackward2<Dtype><<<CAFFE_GET_BLOCKS(dim),
            CAFFE_CUDA_NUM_THREADS>>>( dim, n_buff[i], top_diff+i*dim, 
            bottom_data+i*dim, bottom_diff+i*dim);
        CUDA_POST_KERNEL_CHECK;
      }else{
        Dtype q = p_data[c] / (p_data[c]-1.);
        Dtype qnorm = n_buff[i];
        Dtype inpxdiff;
        caffe_gpu_dot(dim, bottom_data+i*dim, top_diff+i*dim, &inpxdiff);
        // NOLINT_NEXT_LINE(whitespace/operators)
        ConjugateNormBackward<Dtype><<<CAFFE_GET_BLOCKS(dim),
            CAFFE_CUDA_NUM_THREADS>>>(
            dim, q, qnorm, inpxdiff, top_diff+i*dim, 
            bottom_data+i*dim, bottom_diff+i*dim);
        CUDA_POST_KERNEL_CHECK;
      }
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ConjugateNormLayer);


}  // namespace caffe
