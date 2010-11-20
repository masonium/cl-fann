(in-package :fann)

(defun xor-test ()
  "Train a 2,1 standard neural network on a loaded XOR training data set."
  (let ((data (read-train-data-from-file "/home/mason/workspace/fann/xor.net"))
	(nn (create-neural-network '(2 5 1))))
    (setf (activation-function-hidden nn) :sigmoid-symmetric)
    (setf (activation-function-output nn) :sigmoid-symmetric)
    (train-on-data nn data 1000 0 1.0e-6)
    (mapcar (curry #'run nn) '((1.0d0 1.0d0) (-1.0d0 1.0d0) 
			       (1.0d0 -1.0d0) (-1.0d0 -1.0d0)
			       (0.25d0 -0.25d0)
			       (0.1d0 0.7d0)))))

(defun xor2-test ()
  (let ((nn (create-neural-network '(2 1 1 1) :type :shortcut))
	(data (read-train-data-from-file "/home/mason/workspace/fann/hello.txt")))
    (setf (activation-function-hidden nn) :sigmoid-symmetric
	  (activation-function-output nn) :sigmoid-symmetric)
    (fann:cascade-train-on-data nn data 200 1.0e-6)
    (format t "Results on test: ~A~%" (test-on-data nn data))
    (format t "Finished with ~A neurons~%" (total-neurons nn))))