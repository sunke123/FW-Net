#ifndef CAFFE_HOLDER_POOL_LAYER_HPP_
#define CAFFE_HOLDER_POOL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class HolderPoolLayer : public NeuronLayer<Dtype> {
 public:

  explicit HolderPoolLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HolderPool"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype d;
  bool channel_shared_;
  bool fc;
  int channels;
  int dim;
  int group;
  float max_p;
  float min_p;
  bool exp_p;
  Blob<Dtype> forward_buff_1;
  Blob<Dtype> forward_buff_2;
  Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
};

}  // namespace caffe

#endif  // CAFFE_HOLDER_POOL_LAYER_HPP_
