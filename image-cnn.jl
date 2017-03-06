ENV["MOCHA_USE_NATIVE_EXT"] = "true"

using Mocha

backend = CPUBackend()
init(backend)
img_width, img_height, img_channels = (256, 256, 3)
crop_size = (227, 227)
batch_size = 1  # could be larger if you want to classify a bunch of images at a time

layers = [
  MemoryDataLayer(name="data", tops=[:data], batch_size=batch_size,
      transformers=[(:data, DataTransformers.Scale(scale=255)),
                    (:data, DataTransformers.SubMean(mean_file="./model/ilsvrc12_mean.hdf5"))],
      data = Array[zeros(img_width, img_height, img_channels, batch_size)])
  CropLayer(name="crop", tops=[:cropped], bottoms=[:data], crop_size=crop_size)
  ConvolutionLayer(name="conv1", tops=[:conv1], bottoms=[:cropped],
      kernel=(11,11), stride=(4,4), n_filter=96, neuron=Neurons.ReLU())
  PoolingLayer(name="pool1", tops=[:pool1], bottoms=[:conv1],
      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
  LRNLayer(name="norm1", tops=[:norm1], bottoms=[:pool1],
      kernel=5, scale=0.0001, power=0.75)
  ConvolutionLayer(name="conv2", tops=[:conv2], bottoms=[:norm1],
      kernel=(5,5), pad=(2,2), n_filter=256, n_group=2, neuron=Neurons.ReLU())
  PoolingLayer(name="pool2", tops=[:pool2], bottoms=[:conv2],
      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
  LRNLayer(name="norm2", tops=[:norm2], bottoms=[:pool2],
      kernel=5, scale=0.0001, power=0.75)
  ConvolutionLayer(name="conv3", tops=[:conv3], bottoms=[:norm2],
      kernel=(3,3), pad=(1,1), n_filter=384, neuron=Neurons.ReLU())
  ConvolutionLayer(name="conv4", tops=[:conv4], bottoms=[:conv3],
      kernel=(3,3), pad=(1,1), n_filter=384, n_group=2, neuron=Neurons.ReLU())
  ConvolutionLayer(name="conv5", tops=[:conv5], bottoms=[:conv4],
      kernel=(3,3), pad=(1,1), n_filter=256, n_group=2, neuron=Neurons.ReLU())
  PoolingLayer(name="pool5", tops=[:pool5], bottoms=[:conv5],
      kernel=(3,3), stride=(2,2), pooling=Pooling.Max())
  InnerProductLayer(name="fc6", tops=[:fc6], bottoms=[:pool5],
      output_dim=4096, neuron=Neurons.ReLU())
  InnerProductLayer(name="fc7", tops=[:fc7], bottoms=[:fc6],
      output_dim=4096, neuron=Neurons.ReLU())
  InnerProductLayer(name="fc8", tops=[:fc8], bottoms=[:fc7],
      output_dim=1000)
  SoftmaxLayer(name="prob", tops=[:prob], bottoms=[:fc8])
]

net = Net("imagenet", backend, layers)
using HDF5
h5open("model/bvlc_reference_caffenet.hdf5", "r") do h5
  load_network(h5, net)
end
init(net)
classes = open("synset_words.txt") do s map(x -> replace(strip(x), r"^n[0-9]+ ", ""), readlines(s)) end
include("image-classifier.jl")
classifier = ImageClassifier(net, :prob, channel_order=(3,2,1), classes=classes)

using Images, ImageView, Gadfly
img = load("images/cat256.jpg")
imshow(img)
prob, class = classify(classifier, img)
println(class)
n_plot = 5
n_best = sortperm(vec(prob), rev=true)[1:n_plot]
best_probs = prob[n_best]
best_labels = classes[n_best]
plot(x=1:length(best_probs), y=best_probs, color=best_labels, Geom.bar, Guide.ylabel("probability"))