(cl:in-package :cl-user)

(defpackage :cl-fann
  (:nicknames :fann)
  (:use :common-lisp :fann-internal :alexandria)
  (:documentation 
   "CL-FANN is a CFFI wrapper and interface for the libfann artificial
   neural network library.")
  (:export 
   ;; library-loading
   load-fann close-fann

   ;; neural networks
   neural-network neural-network-error create-neural-network run 

   activation-function-hidden activation-function-output
   activation-function-layer activation-steepness-hidden
   activation-steepness-output activation-steepness-layer

   num-input num-output total-neurons total-connections

   training-algorithm learning-rate learning-momentum 
   train-error-function train-stop-function 
   bit-fail-limit 
   quickprop-decay quickprop-mu 
   rprop-increase-factor rprop-decrease-factor rprop-delta-min rprop-delta-max

   load-ann-from-file
   save-ann-to-file
   
   randomize-weights

   ;; training
   copy-train-data merge-train-data subset-train-data read-train-data-from-file
   length-train-data format-train-data scale-train-data shuffle-train-data
   train test train-on-data train-epoch test-on-data init-weights mse reset-mse

   ;; cascade training
   cascade-train-on-data cascade-train-on-file cascade-train
   
   cascade-ouptut-change-fraction cascade-output-stagnation-epochs
   cascade-candidate-change-fraction cascade-candidate-stagnation-epochs
   cascade-weight-multiplier cascade-candidate-limit
   cascade-max-out-epochs cascade-max-cand-epochs
   cascade-activation-functions cascade-activation-steepnesses
   cascade-num-candidate-groups

   cascade-num-candidates cascade-activation-functions-count
   cascade-activation-steepnesses-count 
   ))