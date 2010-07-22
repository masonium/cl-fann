;;;; libfann

(in-package :fann)

(defclass neural-network ()
  ((raw-pointer :initform nil
		:initarg :raw-pointer)
   (num-inputs :initarg :num-inputs)
   (num-outputs :initarg :num-outputs)))

(defmethod print-object ((object neural-network) stream)
  (with-slots (num-inputs num-outputs) object
    (print-unreadable-object (object stream :type t :identity t)
      (format stream "~A,~A" num-inputs num-outputs)))
  object)
			   
(defun make-neural-network-using (init-fn &rest args)
  (let* ((pointer (apply init-fn args))
	 (nn (make-instance 'neural-network 
			    :raw-pointer pointer
			    :num-inputs (fann-get-num-input pointer)
			    :num-outputs (fann-get-num-output pointer))))
    (tg:finalize nn
		 #'(lambda (obj) 
		     (fannint:fann-destroy (slot-value obj 'raw-pointer))))
    nn))

(defun sequence->foreign-array (seq type)
  (cffi:foreign-alloc type :initial-contents seq))

(defmacro with-sequence-as-foreign-array ((var sequence type) &body body)
  `(let ((,var (sequence->foreign-array ,sequence ,type)))
     (unwind-protect 
	  (progn ,@body)
       (cffi:foreign-free ,var))))

(defun create-neural-network (layer-sizes &key (connection-rate 0.1) (type :standard))
  "Create a neural network. :TYPE must be one of :STANDARD, :SPARSE, or :SHORTCUT. 
:CONNECTION-RATE rate is used only for :SPARSE matrices."
  (with-sequence-as-foreign-array (layers layer-sizes :int)
    (let ((num-layers (length layer-sizes)))
      (ecase type
	(:standard
	 (make-neural-network-using 
	  #'fann-internal:fann-create-standard-array 
	  num-layers layers))
	(:sparse
	 (make-neural-network-using 
	  #'fann-internal:fann-create-sparse-array
	  connection-rate
	  num-layers layers))
	(:shortcut
	 (make-neural-network-using
	  #'fann-internal:fann-create-shortcut-array
	  num-layers layers))))))

(defun run (nn input)
  "Run the neural network NN on a sequence INPUT."
  (with-sequence-as-foreign-array (input-array input 'fann-internal:fann-type)
    (let ((output (fann-internal:fann-run (slot-value nn 'raw-pointer) input-array)))
      (prog1 (loop for i from 0 to (1- (slot-value nn 'num-outputs))
		  collecting (cffi:mem-aref output 'fann-internal:fann-type i))
	(cffi:foreign-free output)))))

(defun print-connections (nn)
  (with-slots (raw-pointer) nn
    (fann-internal:fann-print-connections raw-pointer)))
  