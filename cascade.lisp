;;;; cascade.lisp
;;;; Training for cascade neural networks
(in-package :fann)

(defun cascade-train-on-data (nn data max-neurons desired-error)
  "Perform cascade training on a shortcut neural network"
  (fann-internal:fann-cascadetrain-on-data 
   (%pointer nn) (%pointer data) max-neurons 0 (coerce desired-error 'float))) 

(defun cascade-train-on-file (nn pathname max-neurons desired-error)
  "Perform cascade training on a shortcut NN using data from a training dataset"
  (cffi:with-foreign-string (name (namestring pathname))
    (fann-internal:fann-cascadetrain-on-file
     (%pointer nn) name max-neurons 0 (coerce desired-error 'float))))

(defun cascade-train (nn data-spec max-neurons desired-error)
  "Perform cascade trianing on a shortcut NN, using either a loaded training dataset or a filename"
  (funcall (etypecase data-spec
	     (train-data #'cascade-train-on-data)
	     ((or string pathname) #'cascade-train-on-file))
	   nn data-spec max-neurons desired-error))

;;;; read-write accesors
(define-nn-accessors cascade-output-change-fraction cascade-output-stagnation-epochs 
  cascade-candidate-change-fraction cascade-candidate-stagnation-epochs
  cascade-weight-multiplier cascade-candidate-limit
  cascade-max-out-epochs cascade-max-cand-epochs
  cascade-activation-functions cascade-activation-steepnesses
  cascade-num-candidate-groups)

(define-nn-get-accessors cascade-num-candidates cascade-activation-functions-count
  cascade-activation-steepnesses-count )