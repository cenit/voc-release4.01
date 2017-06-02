function [dets, boxes, info, sp, xyls] = imgdetect(input, model, modelinfo, priorinfo, bbox, overlap, pyra, filepath)

% Wrapper that computes detections in the input image.
%
% input    input image
% model    object model
% thresh   detection score threshold
% bbox     ground truth bounding box
% overlap  overlap requirement
% pyra     pyramid
% filepath path to save pyramid/model
% Modified by Haroon Idrees, 2013
% Please do not distribute.

if isempty(pyra)% && strcmp(model.note, 'rc16')==1
    input = color(input);
    tic;
    pyra = featpyramid(input, model);
    save(sprintf('%spyra.mat', filepath), 'pyra', '-v7.3');
    fprintf('generated & saved pyramid in %4.2f seconds\n', toc);
end

if nargin < 4
    bbox = [];
end

if nargin < 5
    overlap = 0;
end

[dets, boxes, info, sp, xyls] = gdetect(pyra, model, modelinfo, priorinfo, bbox, overlap, filepath);
if isempty(dets), return; end;

boxes=reduceboxes(model, boxes);

% map all to component 2
% ---2---
% -6---4-
% -8---7-
% ---5---
% -3---9-
ind2flip=boxes(:,37)==1;
indnoflip=boxes(:,37)==2;
oldboxes=boxes(:,5:36); box1=boxes(:,1:4);
oldsp=sp(:,2:9); sp1=sp(:,1);

boxes=[]; sp=[];
%2
boxes(:,1:4)=oldboxes(:,1:4);
sp(:,1)=oldsp(:,1);
%3
boxes(indnoflip,5:8)=oldboxes(indnoflip,5:8); boxes(ind2flip,5:8)=oldboxes(ind2flip,29:32);
sp(indnoflip,2)=oldsp(indnoflip,2); sp(ind2flip,2)=oldsp(ind2flip,8);
%4
boxes(indnoflip,9:12)=oldboxes(indnoflip,9:12); boxes(ind2flip,9:12)=oldboxes(ind2flip,17:20);
sp(indnoflip,3)=oldsp(indnoflip,3); sp(ind2flip,3)=oldsp(ind2flip,5);
%5
boxes(:,13:16)=oldboxes(:,13:16);
sp(:,4)=oldsp(:,4);
%6
boxes(indnoflip,17:20)=oldboxes(indnoflip,17:20); boxes(ind2flip,17:20)=oldboxes(ind2flip,9:12);
sp(indnoflip,5)=oldsp(indnoflip,5); sp(ind2flip,5)=oldsp(ind2flip,3);
%7
boxes(indnoflip,21:24)=oldboxes(indnoflip,21:24); boxes(ind2flip,21:24)=oldboxes(ind2flip,25:28);
sp(indnoflip,6)=oldsp(indnoflip,6); sp(ind2flip,6)=oldsp(ind2flip,7);
%8
boxes(indnoflip,25:28)=oldboxes(indnoflip,25:28); boxes(ind2flip,25:28)=oldboxes(ind2flip,21:24);
sp(indnoflip,7)=oldsp(indnoflip,7); sp(ind2flip,7)=oldsp(ind2flip,6);
%9
boxes(indnoflip,29:32)=oldboxes(indnoflip,29:32); boxes(ind2flip,29:32)=oldboxes(ind2flip,5:8);
sp(indnoflip,8)=oldsp(indnoflip,8); sp(ind2flip,8)=oldsp(ind2flip,2);

dets(:,5)=2; boxes(:,33)=2;
boxes=[box1, boxes, dets(:,6)]; sp=[sp1, sp];
