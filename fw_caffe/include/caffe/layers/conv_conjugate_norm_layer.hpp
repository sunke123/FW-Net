#ifndef CAFFE_CONV_CONJUGATE_NORM_LAYER_HPP_
#define CAFFE_CONV_CONJUGATE_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvConjugateNormLayer : public NeuronLayer<Dtype> {
 public:

  explicit ConvConjugateNormLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ConvConjugateNorm"; }

 protected:
  //void ConjugateNormLayer<Dtype>::PNorm(const Dtype* inp, 
  //    Dtype* buffer, const Dtype p);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool channel_shared_;
  int dim;
  int group;
  bool spatial;
  int k_size;
  int sp_dim;
  float max_p;
  float min_p;
  bool exp_p;
  //Blob<Dtype> buffer_;         // if necessary, reshape input.
  Blob<Dtype> forward_buff_;   // temporary buffer for forward computation
  Blob<Dtype> norm_buff_;      // store normalization
  Blob<Dtype> norm_buff_2;      // temporary buffer for backward computation 
  Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
  Blob<Dtype> backward_buff_2;  // temporary buffer for backward computation
  Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params

  /// norm_inp stores the input from the ConjugateNormLayer.
  Blob<Dtype> norm_inp_;
  /// norm_opt stores the output from the ConjugateNormLayer.
  Blob<Dtype> norm_opt_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_CONJUGATE_NORM_LAYER_HPP_
