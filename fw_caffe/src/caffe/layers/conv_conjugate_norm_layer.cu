#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conv_conjugate_norm_layer.hpp"

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
  denom = pow(denom, 1./p) + Dtype(1e-10);
  n_buff = denom;
  caffe_gpu_scale(count, Dtype(1./denom), inp, out);
}

template <typename Dtype>
__global__ void NormScaleOut(const bool spatial, const int k_size, 
    const int count, const int channels, const int height, 
    const int width, const Dtype* inp, const Dtype* n_buff, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count){
    if (spatial){
      int num_H = ceil(float(height) / k_size);
      int num_W = ceil(float(width) / k_size);
      int n = index / (channels*height*width);
      int c = index / (height*width) % channels;
      int h = index / width % height;
      int w = index % width;
      int ch = floor(float(h) / k_size);
      int cw = floor(float(w) / k_size);
      int ph = h % k_size;
      int pw = w % k_size;
      int inp_ind = ((((n*channels+c)*num_H+ch)*num_W+cw)*k_size+ph)*k_size+pw;
      int out_ind = ((n*channels+c)*height+h)*width+w;
      out[out_ind] = inp[inp_ind] / n_buff[((n*channels+c)*num_H+ch)*num_W+cw];
    }else{
      int dim = height*width;
      int n = index / (channels*height*width);
      int c = index / (height*width) % channels;
      int h = index / width % height;
      int w = index % width;
      int inp_ind = (n*dim+h*width+w)*channels+c;
      int out_ind = (n*channels+c)*dim+h*width+w;
      out[out_ind] = inp[inp_ind] / n_buff[n*dim+h*width+w];
    }
  }
}

// CUDA kernel for im2col
template <typename Dtype>
__global__ void im2colgpu(const bool spatial, const int k_size, 
    const int count, const int channels, const int height, 
    const int width, const Dtype* inp, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    if (spatial){
      int num_H = ceil(float(height) / k_size);
      int num_W = ceil(float(width) / k_size);
      int n = index / (channels*height*width);
      int c = index / (height*width) % channels;
      int h = index / width % height;
      int w = index % width;
      int ch = floor(float(h) / k_size);
      int cw = floor(float(w) / k_size);
      int ph = h % k_size;
      int pw = w % k_size;
      int out_ind = ((((n*channels+c)*num_H+ch)*num_W+cw)*k_size+ph)*k_size+pw;
      int inp_ind = ((n*channels+c)*height+h)*width+w;
      out[out_ind] = inp[inp_ind];
    }else{
      int dim = height*width;
      int n = index / (channels*height*width);
      int c = index / (height*width) % channels;
      int h = index / width % height;
      int w = index % width;
      int out_ind = (n*dim+h*width+w)*channels+c;
      int inp_ind = (n*channels+c)*dim+h*width+w;
      out[out_ind] = inp[inp_ind];
    }
  }
}

// CUDA kernel for im2col
template <typename Dtype>
__global__ void col2imgpu(const bool spatial, const int k_size, 
    int count, const int channels, const int height, 
    const int width, const Dtype* inp, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    if (spatial){
      int num_H = ceil(float(height) / k_size);
      int num_W = ceil(float(width) / k_size);
      int n = index / (channels*height*width);
      int c = index / (height*width) % channels;
      int h = index / width % height;
      int w = index % width;
      int ch = floor(float(h) / k_size);
      int cw = floor(float(w) / k_size);
      int ph = h % k_size;
      int pw = w % k_size;
      int inp_ind = ((((n*channels+c)*num_H+ch)*num_W+cw)*k_size+ph)*k_size+pw;
      int out_ind = ((n*channels+c)*height+h)*width+w;
      out[out_ind] = inp[inp_ind];
    }else{
      int dim = height*width;
      int n = index / (channels*height*width);
      int c = index / (height*width) % channels;
      int h = index / width % height;
      int w = index % width;
      int inp_ind = (n*dim+h*width+w)*channels+c;
      int out_ind = (n*channels+c)*dim+h*width+w;
      out[out_ind] = inp[inp_ind];
    }
  }
}

template <typename Dtype>
void ConvConjugateNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  Dtype* p_data = this->blobs_[0]->mutable_cpu_data();
  Dtype* n_buff = norm_buff_.mutable_gpu_data();
  Dtype* n_buff_diff = norm_buff_.mutable_gpu_diff();
  Dtype* f_buff = forward_buff_.mutable_gpu_data();
  Dtype* norm_inp = norm_inp_.mutable_gpu_data();
  Dtype* norm_inp_diff = norm_inp_.mutable_gpu_diff();
  Dtype* norm_opt = norm_opt_.mutable_gpu_data();

  im2colgpu<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>( spatial, k_size, count, 
    channels, bottom[0]->height(), bottom[0]->width(), bottom_data, norm_inp);
  CUDA_POST_KERNEL_CHECK;

  if (channel_shared_){
    //LOG(INFO) << "shared_norm!";
    Dtype p = p_data[0];
    if (exp_p){
      p = exp(p);
    }
    if (p > max_p){
      p = max_p;
      p_data[0] = max_p;
      if (exp_p){
        p_data[0] = log(max_p);
      }
    }
    if (p < min_p){
      p = min_p;
      p_data[0] = min_p;
      if (exp_p){
        p_data[0] = log(min_p);
      }
    }
    Dtype q = p / (p - 1.);
    caffe_gpu_abs(count, norm_inp, norm_opt);
    caffe_gpu_powx(count, norm_opt, Dtype(q), norm_inp_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num*dim, sp_dim, 1.,
        norm_inp_.gpu_diff(), multiplier_.gpu_data(), 0.,
        n_buff_diff);
    caffe_gpu_powx(num*dim, n_buff_diff, Dtype(1./q), n_buff);
    caffe_gpu_add_scalar(num*dim, Dtype(1e-10), n_buff);
    //for(int i=0; i<5; i++){
    //  LOG(INFO) << "n_buff: "<< norm_buff_.cpu_data()[i];
    //}
    NormScaleOut<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(spatial, k_size, 
      count, channels, bottom[0]->height(), bottom[0]->width(), norm_inp, n_buff, top_data);
    CUDA_POST_KERNEL_CHECK;
  }else{
    const int div_factor = channel_shared_ ? dim : 1;
    for (int i = 0; i < num*dim; ++i) {
      int c = i % dim / div_factor;
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
      if (p < 1.25){
        n_buff[i] = FindMax(sp_dim, norm_inp+i*dim);
        caffe_gpu_scale(sp_dim, Dtype(1./n_buff[i]), norm_inp, norm_opt);
      }else{
        Dtype q = p / (p - 1.);
        PNorm(sp_dim, q, norm_inp+i*sp_dim, f_buff, n_buff[i], norm_opt+i*sp_dim);
      }
    }
    col2imgpu<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>( spatial, k_size, 
      count, channels, bottom[0]->height(), bottom[0]->width(), norm_opt, top_data);
    CUDA_POST_KERNEL_CHECK;
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
__global__ void ConjugateNormSharedBackward(const int n, const int channels, 
    const Dtype p, const Dtype* pnorm, const Dtype* inpxdiff, const Dtype* in_diff, 
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = index / channels;
    out_diff[index] = in_diff[index] / pnorm[c] - inpxdiff[c] * in_data[index] 
                           * pow((abs(in_data[index])+Dtype(1e-20)),(p-2)) / pow((pnorm[c]+Dtype(1e-10)), (p+1));
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void ConjugateNormBackward(const int n, const Dtype p,
    const Dtype pnorm, const Dtype inpxdiff, const Dtype* in_diff, 
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] / pnorm - inpxdiff * in_data[index]
                           * pow((abs(in_data[index])+Dtype(1e-20)),(p-2)) / pow((pnorm+Dtype(1e-10)), (p+1));
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void ConjugateNormParamBackward2(const int n, 
    const Dtype pnorm, const Dtype m, const Dtype exp_diff,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] += in_diff[index] * in_data[index] / pnorm * m * exp_diff;
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void ConjugateNormParamBackward1(const int n, const int channels,
    const Dtype exp_diff, const Dtype* pnorm, const Dtype* m, const Dtype* in_diff, 
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = index / channels;
    out_diff[index] = in_diff[index] * in_data[index] / pnorm[c] * m[c] * exp_diff;
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void CalculateS3(const int n, const int channels, 
    const Dtype* inp_1, const Dtype* inp_2, const Dtype* s1, 
    const Dtype* s2, Dtype* s3) {
  CUDA_KERNEL_LOOP(index, n) {
    for(int c=0; c<channels; c++){
      s3[index] += inp_1[index*channels+c]*inp_2[index*channels+c];
    }
    s3[index] = s1[index] + s2[index]*s3[index];
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void CalInpXDiff(const int n, const int channels, 
    const Dtype* inp_1, const Dtype* inp_2, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    for(int c=0; c<channels; c++){
      out[index] += inp_1[index*channels+c]*inp_2[index*channels+c];
    }
  }
}

template <typename Dtype>
void ConvConjugateNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* norm_inp = norm_inp_.gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  const Dtype* n_buff = norm_buff_.gpu_data();
  Dtype* n_diff = norm_buff_.mutable_gpu_diff();
  Dtype* norm_inp_diff = norm_inp_.mutable_gpu_diff();
  Dtype* norm_opt_diff = norm_opt_.mutable_gpu_diff();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();

  im2colgpu<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(spatial, k_size, 
    count, channels, bottom[0]->height(), bottom[0]->width(), top_diff, norm_opt_diff);
  CUDA_POST_KERNEL_CHECK;

  int div_factor = channel_shared_ ? dim : 1;

  // Propagate to param
  if (this->param_propagate_down_[0]) {
    Dtype* b_buff_data = backward_buff_.mutable_gpu_data();
    Dtype* b_buff_diff = backward_buff_.mutable_gpu_diff();
    Dtype* n_temp = norm_buff_2.mutable_gpu_data();
    Dtype* n_temp_diff = norm_buff_2.mutable_gpu_diff();
    Dtype* b_diff_2 = backward_buff_2.mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (channel_shared_){
      Dtype* p_diff = this->blobs_[0]->mutable_cpu_diff();
      Dtype p = p_data[0];
      Dtype exp_diff = 1;
      if (exp_p){
        p = exp(p);
        exp_diff = p;
      }
      Dtype q = p / (p-1.);
      // s1
      caffe_gpu_log(num*dim,n_buff,n_diff);
      caffe_gpu_scale(num*dim,Dtype(-(1./p/(p-1.))),n_diff,n_temp);
      // s2
      caffe_gpu_powx(num*dim,n_buff,Dtype(-q),n_diff);
      caffe_gpu_scale(num*dim,Dtype(1./p/(p-1.)),n_diff,n_temp_diff);

      caffe_copy(count, norm_inp, b_buff_data);
      caffe_gpu_abs(count, b_buff_data, b_buff_diff);
      caffe_gpu_add_scalar(count, Dtype(1e-10), b_buff_diff);
      caffe_gpu_log(count, b_buff_diff, norm_inp_diff);
      caffe_gpu_powx(count, b_buff_diff, q, b_buff_data);
      
      // s3 = s1 + s2*s3;
      caffe_gpu_set(num*dim,Dtype(0.),n_diff);
      // NOLINT_NEXT_LINE(whitespace/operators)
      CalculateS3<Dtype><<<CAFFE_GET_BLOCKS(num*dim),
        CAFFE_CUDA_NUM_THREADS>>>(
        num*dim, sp_dim, norm_inp_.gpu_diff(), backward_buff_.gpu_data(), n_temp, n_temp_diff, n_diff);
      CUDA_POST_KERNEL_CHECK;

      //caffe_gpu_set(dim, Dtype(0.), b_buff_diff);
      // NOLINT_NEXT_LINE(whitespace/operators)
      ConjugateNormParamBackward1<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, sp_dim, exp_diff, n_buff, n_diff, norm_opt_diff, norm_inp, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
      Dtype dsum=0;
      caffe_gpu_dot<Dtype>(count, bottom[0]->gpu_diff(), multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), p_diff);
    }
    else{
      Dtype* p_diff = this->blobs_[0]->mutable_gpu_diff();
      int cdim = sp_dim * dim;
      caffe_gpu_set(cdim, Dtype(0.), b_diff_2);
      for (int i=0; i<num*dim; i++){
        int c = i % dim / div_factor;
        Dtype* b_buff_diff_2 = b_diff_2 + (i%dim) *sp_dim;
        Dtype p = p_data[c];
        Dtype exp_diff = 1;
        if (exp_p){
          p = exp(p);
          exp_diff = p;
        }
        Dtype q = p / (p-1.);
        Dtype qnorm = n_buff[i];
        Dtype s1 = -(1./p/(p-1.)) * log(qnorm);
        Dtype s2 = (1./p/(p-1.)) * 1./pow(qnorm,q);
        caffe_copy(sp_dim, norm_inp+i*sp_dim, b_buff_data);
        caffe_gpu_add_scalar(sp_dim, Dtype(1e-20), b_buff_data);
        caffe_gpu_powx(sp_dim, b_buff_data, Dtype(2), b_buff_diff);
        caffe_gpu_powx(sp_dim, b_buff_diff, Dtype(0.5), b_buff_data);
        caffe_gpu_log(sp_dim, b_buff_data, bottom_diff);
        caffe_gpu_powx(sp_dim, b_buff_data, q, b_buff_diff);
        Dtype s3;
        caffe_gpu_dot(sp_dim, bottom[0]->gpu_diff(), backward_buff_.gpu_diff(), &s3);
        s3 = s1 + s2*s3;
        //caffe_gpu_set(dim, Dtype(0.), b_buff_diff);
        // NOLINT_NEXT_LINE(whitespace/operators)
        ConjugateNormParamBackward2<Dtype><<<CAFFE_GET_BLOCKS(sp_dim),
          CAFFE_CUDA_NUM_THREADS>>>(
          sp_dim, qnorm, s3, exp_diff, norm_opt_diff+i*sp_dim, norm_inp+i*sp_dim, b_buff_diff_2);
        CUDA_POST_KERNEL_CHECK;
      }
      caffe_gpu_gemv<Dtype>(CblasNoTrans, dim, sp_dim, 1.,
        backward_buff_2.gpu_diff(), multiplier_.gpu_data(), 1.,
        p_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(count,Dtype(0.),bottom_diff);
    caffe_gpu_set(num*dim,Dtype(0.),n_diff);
    if (channel_shared_){
      Dtype p = p_data[0];
      if (exp_p){
        p = exp(p);
      }
      Dtype q = p / (p-1.);
      // inpxdiff;
      // 1. caffe_gpu_gemm(CblasNoTrans,CblasTrans,)
      // NOLINT_NEXT_LINE(whitespace/operators)
      CalInpXDiff<Dtype><<<CAFFE_GET_BLOCKS(num*dim),
          CAFFE_CUDA_NUM_THREADS>>>(
        num*dim, sp_dim, norm_inp, norm_opt_diff, n_diff);
      CUDA_POST_KERNEL_CHECK;
      // NOLINT_NEXT_LINE(whitespace/operators)
      ConjugateNormSharedBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
          count, sp_dim, q, n_buff, n_diff, norm_opt_diff, 
          norm_inp, norm_inp_diff);
      CUDA_POST_KERNEL_CHECK;
    }else{
      for(int i=0; i<num*dim; i++){
        int c = i % dim / div_factor;
        Dtype p = p_data[c];
        if (exp_p){
          p = exp(p);
        }
        if (p < 1.25){
          ConjugateNormBackward2<Dtype><<<CAFFE_GET_BLOCKS(sp_dim),
              CAFFE_CUDA_NUM_THREADS>>>( sp_dim, n_buff[i], norm_opt_diff+i*sp_dim, 
              norm_inp+i*sp_dim, norm_inp_diff+i*sp_dim);
          CUDA_POST_KERNEL_CHECK;
        }else{
          Dtype q = p / (p-1.);
          Dtype qnorm = n_buff[i];
          Dtype inpxdiff;
          caffe_gpu_dot(sp_dim, norm_inp+i*sp_dim, norm_opt_diff+i*sp_dim, &inpxdiff);
          //LOG(INFO) << "inpxdiff: " << inpxdiff;
          // NOLINT_NEXT_LINE(whitespace/operators)
          ConjugateNormBackward<Dtype><<<CAFFE_GET_BLOCKS(sp_dim),
              CAFFE_CUDA_NUM_THREADS>>>(
              sp_dim, q, qnorm, inpxdiff, norm_opt_diff+i*sp_dim, 
              norm_inp+i*sp_dim, norm_inp_diff+i*sp_dim);
          CUDA_POST_KERNEL_CHECK;
        }
      }
    }
    col2imgpu<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(spatial, k_size,
        count, channels, bottom[0]->height(), bottom[0]->width(), norm_inp_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ConvConjugateNormLayer);


}  // namespace caffe
