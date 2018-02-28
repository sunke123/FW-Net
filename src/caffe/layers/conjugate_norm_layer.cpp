#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conjugate_norm_layer.hpp"

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
void ConjugateNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ConjugateNormParameter norm_param = this->layer_param().conjugate_norm_param();
  int channels = bottom[0]->channels();
  channel_shared_ = norm_param.channel_shared();
  fc = norm_param.fc();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
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
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "p size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  forward_buff_.Reshape(vector<int>(1, bottom[0]->count(2)));
  vector<int> norm_size(1,bottom[0]->num());
  norm_size.push_back(bottom[0]->channels());
  norm_buff_.Reshape(norm_size);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  if (fc){
    backward_buff_.Reshape(vector<int>(1, bottom[0]->channels()));
  }else{
    backward_buff_.Reshape(vector<int>(1, bottom[0]->count(2)));
  }
  backward_buff_2.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void ConjugateNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ConjugateNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

  const Dtype* p_data = this->blobs_[0]->cpu_data();
  Dtype* f_buff = forward_buff_.mutable_cpu_data();
  Dtype* n_buff = norm_buff_.mutable_cpu_data();

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < num*channels; ++i) {
    int c = i % channels / div_factor;
    Dtype q = p_data[c] / (p_data[c] - 1.);
    PNorm(dim, q, bottom_data, f_buff, n_buff[i], top_data);
    bottom_data += dim;
    top_data += dim;
  }
}

template <typename Dtype>
void ConjugateNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  const Dtype* n_buff = norm_buff_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  if (this->param_propagate_down_[0]) {
    Dtype* p_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* b_buff_data = backward_buff_.mutable_cpu_data();
    Dtype* b_buff_diff = backward_buff_.mutable_cpu_diff();
    for (int i=0; i<num*channels; i++){
      int c = i % channels / div_factor;
      Dtype p = p_data[c];
      Dtype q = p / (p-1.);
      Dtype s1 = -(1./p/(p-1.)) * log(n_buff[i]);
      Dtype s2 = (1./p/(p-1.)) * 1./pow(n_buff[i],q);
      caffe_copy(dim, bottom_data+i*dim, b_buff_data);
      caffe_add_scalar(dim, Dtype(1e-20), b_buff_data);
      caffe_powx(dim, b_buff_data, Dtype(2), b_buff_diff);
      caffe_powx(dim, b_buff_diff, Dtype(0.5),  b_buff_data);
      caffe_log(dim, b_buff_data, b_buff_diff);
      caffe_powx(dim, b_buff_data, q, bottom_diff);
      Dtype s3 = caffe_cpu_dot(dim, b_buff_diff, bottom_diff);;
      s3 = s2*s3;
      for (int j = 0; j < dim; ++j) {
        p_diff[c] += top_diff[i*dim+j] * bottom_data[i*dim+j] / n_buff[i] * (s1+s3);
      }
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    for(int i=0; i<num*channels; i++){
      int c = i % channels / div_factor;
      Dtype q = p_data[c] / (p_data[c]-1.);
      Dtype inpxdiff = caffe_cpu_dot(dim, top_diff+i*dim, bottom_data+i*dim);
      for(int j=0; j<dim; j++){
        bottom_diff[i*dim+j] = top_diff[i*dim+j] / n_buff[i] - 
                              inpxdiff * bottom_data[i*dim+j]
                              * pow((abs(bottom_data[i*dim+j])+Dtype(1e-20)),(q-2)) / pow((n_buff[i]+Dtype(1e-10)),(q+1));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ConjugateNormLayer);
#endif

INSTANTIATE_CLASS(ConjugateNormLayer);
REGISTER_LAYER_CLASS(ConjugateNorm);

}  // namespace caffe
