;;;; train.lisp
;;;; Training functions and data

(in-package :fann)

(defclass train-data ()
  ((raw-pointer :initform nil
		:initarg :raw-pointer)
   (num-inputs)
   (num-outputs)
   (length)))

(defmethod print-object ((object train-data) stream)
  (with-slots (num-inputs num-outputs) object
    (print-unreadable-object (object stream :type t :identity t)
      (format stream "data")))
  object)
			   
(defun %make-train-data (pointer)
  "Create a training data from a raw pointer"
  (let* ((nn (make-instance 'train-data 
			    :raw-pointer pointer
			    :num-inputs (fann-num-input-train-data pointer)
			    :num-outputs (fann-num-output-train-data pointer)
			    :length (fann-length-train-data pointer))))
	   
    (tg:finalize nn
		 #'(lambda (obj) 
		     (fannint:fann-destroy-train (slot-value obj 'raw-pointer))))
    nn))

(defun %data-pointer (train-data)
  "Returns the raw pointer to the internal fann_train_data object"
  (slot-value train-data 'raw-pointer))

(defun read-train-data-from-file (pathname)
  "Create a training set with data loaded from PATHNAME"
  (cffi:with-foreign-string (data-filename (namestring pathname))
    (%make-train-data 
     (fann-read-train-from-file data-filename))))

(defun format-train-data (stream inputs outputs)
  "Write a training set to STREAM in the format that FANN can load. INPUTS is a list of lists, with each constituent list an input vector. If the inputs are dnimension one, the INPUTS can simply be a list of numbers, rather than a list of singletons. OUTPUTS is similarly formatted. If INPUTS and OUTPUTS are of different sizes, the functions stops once the shorter one runs out."
  (let ((N (length inputs)))
    (format stream "~A~%" N)
    (mapc 
     #'(lambda (in out)
	 (format stream "~{~A~^ ~}~%~{~A~^ ~}~%" in out))
     inputs outputs)))

(defun scale-train-data (train-data new-min new-max &optional (input t) (output t))
  "Scale the inputs and/or outputs in TRAIN-DATA to the range (NEW-MIN, NEW-MAX)"
  (let ((pointer (%data-pointer train-data)))
    (funcall 
     (cond 
       ((and input output) #'fann-scale-train-data pointer)
       (input #'fann-scale-input-train-data)
       (output #'fann-scale-output-train-data)
       (t #'identity))
     new-min new-max)))

(defun shuffle-train-data (train-data)
  "Shuffle the training data to be in a random order"
  (fann-shuffle-train-data (%data-pointer train-data)))

(defun train (nn input desired-output)
  "Train NN on a single (INPUT, DESIRE-OUTPUT) data pair"
  (with-sequence-as-foreign-array (in input 'fann-internal:fann-type
				      out desired-output 'fann-internal:fann-type)
    (fann-train (%nn-pointer nn) in out)))

;;;; training parameters
