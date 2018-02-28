#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/holder_pool_layer.hpp"

namespace caffe {

template <typename Dtype>
void HolderPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  HolderPoolParameter pool_param = this->layer_param().holder_pool_param();
  channel_shared_ = pool_param.channel_shared();
  fc = pool_param.fc();
  d = pool_param.d();
  group = pool_param.group();
  max_p = pool_param.max_p();
  min_p = pool_param.min_p();
  exp_p = pool_param.exp_p();
  int channels = bottom[0]->channels();
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
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (pool_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(pool_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.5);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  vector<int> norm_size(1,bottom[0]->num());
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), group)
        << "p size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "p size is inconsistent with prototxt config";
    norm_size.push_back(bottom[0]->channels());
  }
  
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void HolderPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (fc){
    forward_buff_1.Reshape(vector<int>(1, bottom[0]->count(1)));
    forward_buff_2.Reshape(vector<int>(1, bottom[0]->count(1)));
  }else{
    forward_buff_1.Reshape(vector<int>(1, bottom[0]->count()));
    forward_buff_2.Reshape(vector<int>(1, bottom[0]->count()));    
  }
  multiplier_.Reshape(vector<int>(1, bottom[0]->count()));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void HolderPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  //const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }
  Dtype* p_data = this->blobs_[0]->mutable_cpu_data();
  Dtype* f_data_1 = forward_buff_1.mutable_cpu_data();
  Dtype* f_diff_1 = forward_buff_1.mutable_cpu_diff();
  Dtype* f_data_2 = forward_buff_2.mutable_cpu_data();

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels/group : 1;
  for (int i = 0; i < num*channels; ++i) {
    int c = i % channels / div_factor;
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
    Dtype q = 1. / (p-1);
    caffe_abs(dim, bottom_data, f_data_1);
    caffe_powx(dim, f_data_1, q, f_diff_1);
    caffe_powx(dim, bottom_data, Dtype(2), f_data_1);
    caffe_add_scalar(dim, Dtype(1e-20), f_data_1);
    caffe_powx(dim, f_data_1, Dtype(0.5), f_data_2);
    caffe_div(dim, bottom_data, f_data_2, f_data_1);
    caffe_mul(dim, f_data_1, f_diff_1, f_data_2);
    caffe_cpu_scale(dim, -d, f_data_2, top_data);
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Dtype>
void HolderPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels/group : 1;

  // Propagte to param
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* p_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* b_buff_data = backward_buff_.mutable_cpu_data();
    Dtype* b_buff_diff = backward_buff_.mutable_cpu_diff();
    for (int i=0; i<num*channels; i++){
      int c = i % channels / div_factor;
      Dtype p = p_data[c];
      Dtype exp_diff = 1;
      if (exp_p){
        p = exp(p);
        exp_diff = p;
      }
      Dtype q = 1. / (p-1.);
      caffe_powx(dim, bottom_data, Dtype(2), b_buff_diff);
      caffe_add_scalar(dim, Dtype(1e-20), b_buff_diff);
      caffe_powx(dim, b_buff_diff, Dtype(0.5), b_buff_data);
      
      caffe_powx(dim, b_buff_data, q, bottom_diff);
      caffe_mul(dim, bottom_diff, bottom_data, b_buff_diff);
      caffe_div(dim, b_buff_diff, b_buff_data, bottom_diff);
      for (int j = 0; j < dim; ++j) {
        p_diff[c] += d * (q*q) * top_diff[j] * 
                  bottom_diff[j] * log(abs(bottom_data[j]))*exp_diff;
      }
      top_diff += dim;
      bottom_data += dim;
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* b_buff_data = backward_buff_.mutable_cpu_data();
    Dtype* b_buff_diff = backward_buff_.mutable_cpu_diff();
    for(int i=0; i<num*channels; i++){
      int c = i % channels / div_factor;
      Dtype p = p_data[c];
      if (exp_p){
        p = exp(p);
      }
      Dtype q = 1. / (p-1.);
      caffe_powx(dim, bottom_data, Dtype(2), b_buff_data);
      caffe_add_scalar(dim, Dtype(1e-20), b_buff_data);
      caffe_powx(dim, b_buff_data, Dtype(0.5), b_buff_diff);
      caffe_powx(dim, b_buff_diff, q, b_buff_data);

      caffe_mul(dim, b_buff_data, top_diff, b_buff_diff);

      caffe_powx(dim, bottom_data, Dtype(2), bottom_diff);
      Dtype s1 = p*Dtype(1e-20);
      caffe_add_scalar(dim, s1, bottom_diff);
      caffe_mul(dim, bottom_diff, b_buff_diff, b_buff_data);
      Dtype s2 = Dtype(1e-20) - p*Dtype(1e-20);
      caffe_add_scalar(dim, s2, bottom_diff);
      caffe_powx(dim, bottom_diff, Dtype(1.5), b_buff_diff);
      caffe_div(dim, b_buff_data, b_buff_diff, bottom_diff);
      Dtype s3 = -d / (p-1.);
      caffe_scal(dim, s3, bottom_diff);
      bottom_diff += dim;
      top_diff += dim;
      bottom_data += dim;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(HolderPoolLayer);
#endif

INSTANTIATE_CLASS(HolderPoolLayer);
REGISTER_LAYER_CLASS(HolderPool);

}  // namespace caffe
