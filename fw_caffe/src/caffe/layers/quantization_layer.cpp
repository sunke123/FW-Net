#include <algorithm>
#include <vector>
#include <math.h>

#include "caffe/layers/quantization_layer.hpp"

namespace caffe {

template <typename Dtype>
void QuantizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = round(bottom_data[i]);
  }
}

template <typename Dtype>
void QuantizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(),top_diff,bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(QuantizationLayer);
#endif

INSTANTIATE_CLASS(QuantizationLayer);
REGISTER_LAYER_CLASS(Quantization);

}  // namespace caffe
