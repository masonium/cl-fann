(in-package :fann)

(defun xor-test ()
  "Train a 2,1 standard neural network on a loaded XOR training data set."
  (let ((data (read-train-data-from-file "/home/mason/workspace/fann/xor.net"))
	(nn (create-neural-network '(2 5 1))))
    (setf (activation-function-hidden nn) :sigmoid-symmetric)
    (setf (activation-function-output nn) :sigmoid-symmetric)
    (train-on-data nn data 1000 0 1.0e-4)
    (mapcar (curry #'run nn) '((1.0d0 1.0d0) (-1.0d0 1.0d0) 
			       (1.0d0 -1.0d0) (-1.0d0 -1.0d0)))))
