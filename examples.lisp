(defpackage :fann-examples
  (:use :common-lisp :fann :alexandria :bind))

(in-package :fann-examples)

(defun xor-example ()
  "Train a 2,1 standard neural network on a loaded XOR training data set."
  (let ((data (read-train-data-from-file "/home/mason/workspace/fann/xor.net"))
	(nn (create-neural-network '(2 5 1))))
    (setf (activation-function-hidden nn) :sigmoid-symmetric)
    (setf (activation-function-output nn) :sigmoid-symmetric)
    (train-on-data nn data 1000 0 1.0e-6)
    (mapcar (curry #'run nn) '((1.0d0 1.0d0) (-1.0d0 1.0d0) 
			       (1.0d0 -1.0d0) (-1.0d0 -1.0d0)))))

(defun linear-regression-example ()
  "Create a data-set of points in 3d space, separated by a plane aX +
bY + cZ + D = 0"
  (bind (((a b c d) '(2.0 3.0 1.0 -2.0))
	 (n 10000)
	 ;; create the data
	 (data-inputs (loop 
			 :repeat n 
			 :collect (list (random 1.0) (random 1.0) (random 1.0))))
	 (data-outputs (mapcar #'(lambda (p) (destructuring-bind (x y z) p 
					       (+ (* a x) (* b y) (* c z) d)))
			       data-inputs)))
    (with-output-to-file (str "ls.dat" :if-exists :supersede) 
      (format-train-data str data-inputs data-outputs)))
  (bind ((data (read-train-data-from-file "ls.dat"))
	 (nn (create-neural-network '(3 10 1))))
    (setf (activation-function-hidden nn) :gaussian
	  (activation-function-output nn) :linear)
    (init-weights nn data)
    (train-on-data nn data 10000 0 1e-4)
    (reset-mse nn)
    (test-on-data nn data)))