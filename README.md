## mocha_cnn
image classification with pre-trained imagenet cnn updated for julia v0.5 using mocha

### prerequisite
* [julia v0.5](http://julialang.org/downloads/)
* Mocha.jl, Images.jl, ImageView.jl, Gadfly.jl
```julia
Pkg.add("Mocha")
Pkg.add("Images")
Pkg.add("ImageView")
Pkg.add("Gadfly")
Pkg.test("Mocha")
```
* run `./get-model.sh`

### usage
```
$ julia
julia> include("image-cnn.jl")
```

### remark
* my contribution: update for julia v0.5
* original example: [link](http://nbviewer.jupyter.org/github/pluskid/Mocha.jl/blob/master/examples/ijulia/ilsvrc12/imagenet-classifier.ipynb)
