#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conv_conjugate_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void PNorm(const int count, const Dtype p, 
    const Dtype* inp, Dtype* f_buff, Dtype &n_buff, Dtype* out){
  caffe_abs(count, inp, f_buff);
  caffe_powx(count, f_buff, p, out);
  Dtype denom = caffe_cpu_asum(count, out);
  denom = pow(denom, 1./p);
  n_buff = denom;
  caffe_cpu_scale(count, Dtype(1./denom), inp, out);
}

template <typename Dtype>
void im2col(const bool spatial, const int k_size, const int num, 
  const int channels, const int height, const int width, 
  const Dtype* inp, Dtype* out){
  if (spatial){
    int num_H = ceil(height / k_size);
    int num_W = ceil(width / k_size);
    for (int n=0; n<num; n++){
      for (int c=0; c<channels; c++){
        for (int h=0; h<height; h++){
          for (int w=0; w<width; w++){
            int ch = floor(h / k_size);
            int cw = floor(w / k_size);
            int ph = h % k_size;
            int pw = w % k_size;
            int out_ind = ((((n*channels+c)*num_H+ch)*num_W+cw)*k_size+ph)*k_size+pw;
            int inp_ind = ((n*channels+c)*height+h)*width+w;
            out[out_ind] = inp[inp_ind];
          }
        }
      }
    }
  }else{
    int dim = height*width;
    for (int n=0; n<num; n++){
      for (int h=0; h<height; h++){
        for (int w=0; w<width; w++){
          for (int c=0; c<channels; c++){
            int out_ind = (n*dim+h*width+w)*channels+c;
            int inp_ind = (n*channels+c)*dim+h*width+w;
            out[out_ind] = inp[inp_ind];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void col2im(const bool spatial, const int k_size, const int num, 
  const int channels, const int height, const int width, 
  const Dtype* inp, Dtype* out){
  if (spatial){
    int num_H = ceil(height / k_size);
    int num_W = ceil(width / k_size);
    for (int n=0; n<num; n++){
      for (int c=0; c<channels; c++){
        for (int h=0; h<height; h++){
          for (int w=0; w<width; w++){
            int ch = floor(h / k_size);
            int cw = floor(w / k_size);
            int ph = h % k_size;
            int pw = w % k_size;
            int inp_ind = ((((n*channels+c)*num_H+ch)*num_W+cw)*k_size+ph)*k_size+pw;
            int out_ind = ((n*channels+c)*height+h)*width+w;
            out[out_ind] = inp[inp_ind];
          }
        }
      }
    }
  }else{
    int dim = height*width;
    for (int n=0; n<num; n++){
      for (int h=0; h<height; h++){
        for (int w=0; w<width; w++){
          for (int c=0; c<channels; c++){
            int inp_ind = (n*dim+h*width+w)*channels+c;
            int out_ind = (n*channels+c)*dim+h*width+w;
            out[out_ind] = inp[inp_ind];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void ConvConjugateNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ConjugateNormParameter norm_param = this->layer_param().conjugate_norm_param();
  channel_shared_ = norm_param.channel_shared();
  group = norm_param.group();
  spatial = norm_param.spatial();
  k_size = norm_param.kernel_size();
  max_p = norm_param.max_p();
  min_p = norm_param.min_p();
  exp_p = norm_param.exp_p();
  if (spatial){
    dim = bottom[0]->channels()*ceil(bottom[0]->height() / k_size) 
                                  * ceil(bottom[0]->width() / k_size);
  }else{
    dim = bottom[0]->count(2);
  }
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      if (group==1){
        this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      }else{
        this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1,group)));
      }
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, dim)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (norm_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(norm_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.5);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "p size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), dim)
        << "p size is inconsistent with prototxt config";
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ConvConjugateNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  if (spatial){
    dim = bottom[0]->channels()*ceil(bottom[0]->height() / k_size) 
              * ceil(bottom[0]->width() / k_size);
    sp_dim = k_size*k_size;
  }else{
    dim = bottom[0]->count(2);
    sp_dim = bottom[0]->channels();
  }
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> norm_inp_size(1,bottom[0]->num()*dim);
  norm_inp_size.push_back(sp_dim);
  norm_inp_.Reshape(norm_inp_size);
  norm_opt_.Reshape(norm_inp_size);
  norm_buff_.Reshape(vector<int>(1, bottom[0]->num()*dim));
  norm_buff_2.Reshape(vector<int>(1, bottom[0]->num()*dim));
  multiplier_.Reshape(vector<int>(1, bottom[0]->count()));
  forward_buff_.Reshape(vector<int>(1, sp_dim));
  //backward_buff_.Reshape(vector<int>(1, bottom[0]->channels()));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count()));
  backward_buff_2.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void ConvConjugateNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* norm_inp = norm_inp_.mutable_cpu_data();
  Dtype* norm_opt = norm_opt_.mutable_cpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  Dtype* p_data = this->blobs_[0]->mutable_cpu_data();
  Dtype* f_buff = forward_buff_.mutable_cpu_data();
  Dtype* n_buff = norm_buff_.mutable_cpu_data();

  im2col(spatial, k_size, num, channels, bottom[0]->height(), 
        bottom[0]->width(), bottom_data, norm_inp);

  // if channel_shared, channel index in the following computation becomes
  // always zero.
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
    Dtype q = p / (p - 1.);
    PNorm(sp_dim, q, norm_inp+i*sp_dim, f_buff, n_buff[i], norm_opt+i*sp_dim);
  }
  col2im(spatial, k_size, num, channels, bottom[0]->height(), 
        bottom[0]->width(), norm_opt, top_data);
}

template <typename Dtype>
void ConvConjugateNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* norm_inp = norm_inp_.cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  const Dtype* n_buff = norm_buff_.cpu_data();
  Dtype* norm_inp_diff = norm_inp_.mutable_cpu_diff();
  Dtype* norm_opt_diff = norm_opt_.mutable_cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();

  im2col(spatial, k_size, num, channels, bottom[0]->height(), 
          bottom[0]->width(), top_diff, norm_opt_diff);

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? dim : 1;

  // Propagte to param
  if (this->param_propagate_down_[0]) {
    Dtype* p_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* b_buff_data = backward_buff_.mutable_cpu_data();
    Dtype* b_buff_diff = backward_buff_.mutable_cpu_diff();
    for (int i=0; i<num*dim; i++){
      int c = i % dim / div_factor;
      Dtype p = p_data[c];
      Dtype exp_diff = 1;
      if (exp_p){
        p = exp(p);
        exp_diff = p;
      }
      Dtype q = p / (p-1.);
      Dtype s1 = -(1./p/(p-1.)) * log(n_buff[i]);
      Dtype s2 = (1./p/(p-1.)) * 1./pow(n_buff[i],q);
      caffe_copy(sp_dim, norm_inp+i*sp_dim, b_buff_data);
      caffe_add_scalar(sp_dim, Dtype(1e-20), b_buff_data);
      caffe_powx(sp_dim, b_buff_data, Dtype(2), b_buff_diff);
      caffe_powx(sp_dim, b_buff_diff, Dtype(0.5),  b_buff_data);
      caffe_log(sp_dim, b_buff_data, b_buff_diff);
      caffe_powx(sp_dim, b_buff_data, q, bottom_diff);
      Dtype s3 = caffe_cpu_dot(sp_dim, b_buff_diff, bottom_diff);;
      s3 = s2*s3;
      for (int j = 0; j < channels; ++j) {
        p_diff[c] += norm_opt_diff[i*sp_dim+j] * norm_inp[i*sp_dim+j] / n_buff[i] * (s1+s3) * exp_diff;
      }
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    for(int i=0; i<num*dim; i++){
      int c = i % dim / div_factor;
      Dtype p = p_data[c];
      if (exp_p){
        p = exp(p);
      }
      Dtype q = p / (p-1.);
      Dtype inpxdiff = caffe_cpu_dot(sp_dim, norm_opt_diff+i*sp_dim, norm_inp+i*sp_dim);
      //LOG(INFO) << "inpxdiff: " << inpxdiff;
      for(int j=0; j<channels; j++){
        norm_inp_diff[i*sp_dim+j] = norm_opt_diff[i*sp_dim+j] / n_buff[i] - 
                              inpxdiff * norm_inp[i*sp_dim+j]
                              * pow((abs(norm_inp[i*sp_dim+j])+Dtype(1e-10)),(q-2)) / pow((n_buff[i]+Dtype(0)),(q+1));
      }
    }
    col2im(spatial, k_size, num, channels, bottom[0]->height(), 
              bottom[0]->width(), norm_inp_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ConvConjugateNormLayer);
#endif

INSTANTIATE_CLASS(ConvConjugateNormLayer);
REGISTER_LAYER_CLASS(ConvConjugateNorm);

}  // namespace caffe
