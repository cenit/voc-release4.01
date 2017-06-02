mex -O resize.cc
mex -O dt.cc
mex -O features.cc
mex -O getdetections.cc

% use one of the following depending on your setup
% 0 is fastest, 2 is slowest 

% 0) SSE multithreaded convolution
mex -output fconv -O fconvsse.cc

% 1) mulththreaded convolution
% mex -output fconv -O fconvMT.cc

% 2) basic convolution, very compatible
% mex -output fconv -O fconv.cc
