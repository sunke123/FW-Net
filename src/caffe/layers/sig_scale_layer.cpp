#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/sig_scale_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  SigScaleParameter scale_param = this->layer_param().sig_scale_param();
  int channels = bottom[0]->channels();
  channel_shared_ = scale_param.channel_shared();
  counterpart = scale_param.counterpart();
  fc = scale_param.fc();
  group = scale_param.group();
  CHECK_EQ(channels/group,Dtype(channels)/group) << "Channel should be times of group.";
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
    if (scale_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(scale_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), group)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SigScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SigScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count(2);
  if (fc){
	  channels = 1;
	  dim = bottom[0]->channels();
  }
  const Dtype* p_data = this->blobs_[0]->cpu_data();

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  caffe_copy(count, bottom_data, top_data);
  const int div_factor = channel_shared_ ? channels/group : 1;
  for (int i = 0; i < num*channels; ++i) {
    int c = i % channels / div_factor;
    Dtype r = sigmoid(p_data[c]);
    if (counterpart){
      r = 1 - sigmoid(p_data[c]);
    }
    caffe_scal(dim, r, top_data);
    top_data += dim;
  }
}

template <typename Dtype>
void SigScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* p_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
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
    Dtype* p_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < num*channels; ++i) {
      int c = i % channels / div_factor;
      Dtype r = exp(-p_data[c]) / pow((1+exp(-p_data[c])),2);
      if (counterpart){
        r = -r;
      }
      p_diff[c] += caffe_cpu_dot(dim, bottom_data, top_diff+i*dim)*r;
      bottom_data += dim;
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    for (int i = 0; i < num*channels; ++i) {
      int c = i % channels / div_factor;
      Dtype r = sigmoid(p_data[c]);
      if (counterpart){
        r = 1 - sigmoid(p_data[c]);
      }
      caffe_scal(dim, r, bottom_diff);
      bottom_diff += dim;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SigScaleLayer);
#endif

INSTANTIATE_CLASS(SigScaleLayer);
REGISTER_LAYER_CLASS(SigScale);

}  // namespace caffe
