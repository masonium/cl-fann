;;;; train.lisp
;;;; Training functions and data

(in-package :fann)

(defclass train-data ()
  ((raw-pointer :initform nil
		:initarg :raw-pointer)
   (num-inputs :initarg :num-inputs)
   (num-outputs :initarg :num-outputs)
   (length :initarg :length)))

(defmethod print-object ((object train-data) stream)
  (with-slots (num-inputs num-outputs) object
    (print-unreadable-object (object stream :type t :identity t)
      (format stream "data")))
  object)

;;;; Construction
(defun %make-train-data (pointer)
  "Create a training data from a raw pointer"
  (let* ((train-data (make-instance 'train-data 
				    :raw-pointer pointer
				    :num-inputs (fann-num-input-train-data pointer)
				    :num-outputs (fann-num-output-train-data pointer)
				    :length (fann-length-train-data pointer))))
    
    (tg:finalize train-data
		 #'(lambda () (fannint:fann-destroy-train pointer)))
    train-data))

(defun copy-train-data (data)
  "Make an exact copy of the training set"
  (%make-train-data (fann-internal:fann-duplicate-train-data (%pointer data))))

(defun merge-train-data (data1 data2)
  "Merge DATA1 and DATA2 into a newly-allocated data structure"
  (%make-train-data (fann-internal:fann-merge-train-data 
		     (%pointer data1) (%pointer data2))))

(defun subset-train-data (data &optional (pos 0) (length (length-train-data data)))
  "Return a subset of the training data, starting at POS"
  (let ((len (min (- (length-train-data data) pos) length)))
    (%make-train-data (fann-internal:fann-subset-train-data data pos len))))

(defun read-train-data-from-file (pathname)
  "Create a training set with data loaded from PATHNAME"
  (cffi:with-foreign-string (data-filename (namestring pathname))
    (%make-train-data 
     (fann-read-train-from-file data-filename))))

(defun length-train-data (data)
  "Return the number of examples in DATA"
  (fann-internal:fann-length-train-data (%pointer data )))

(defun format-train-data (stream inputs outputs)
  "Write a training set to STREAM in the format that FANN can
load. INPUTS is a sequence of sequences, with each constituent sequence an input
vector. If the inputs are of dimension one, the INPUTS can simply be a
sequence of numbers, rather than a list of singletons. OUTPUTS is
similarly formatted. If INPUTS and OUTPUTS are of different sizes, the
functions stops once the shorter one runs out."
  (let ((N (min (length inputs) (length outputs))))
    (labels ((convert-data (data)
	       (cond
		 ((listp data) data)
		 ((vectorp data) (coerce data 'list))
		 ((numberp data) (list data))
		 (t (error "Each input must be a number or a sequence of numbers")))))
      (format stream "~A~%" N)
      (map nil
	   #'(lambda (in out)
	       (format stream "~{~A~^ ~}~%~{~A~^ ~}~%" 
		       (convert-data in) 
		       (convert-data out)))
	   inputs
	   outputs))))

(defun scale-train-data (train-data new-min new-max 
			 &optional (input t) (output t))
  "Scale the inputs and/or outputs in TRAIN-DATA to the range (NEW-MIN, NEW-MAX)"
  (let ((pointer (%pointer train-data)))
    (funcall 
     (cond 
       ((and input output) #'fann-scale-train-data pointer)
       (input #'fann-scale-input-train-data)
       (output #'fann-scale-output-train-data)
       (t #'(lambda (x y)
	      (declare (ignore x y)) nil)))
     new-min new-max)))

(defun shuffle-train-data (train-data)
  "Shuffle the training data to be in a random order"
  (fann-shuffle-train-data (%pointer train-data)))

(defun train (nn input desired-output)
  "Train NN on a single (INPUT, DESIRED-OUTPUT) data pair"
  (with-sequence-as-foreign-array (in input 'fann-internal:fann-type 
				      out desired-output 'fann-internal:fann-type)
    (fann-train (%pointer nn) in out)))

(defun test (nn input desired-output)
  "Test NN on a single data pair, updating the internal mse"
  (with-sequence-as-foreign-array (in input 'fann-internal:fann-type
				      out desired-output 'fann-internal:fann-type)
    (fann-test (%pointer nn) in out)))

(defun train-on-data (nn data max-epochs epochs-between-reports desired-error)
  "Train NN on the DATA. DATA can be either a PATHNAME to a file in
the correct data format or a pre-loaded TRAIN-DATA dataset"
  (etypecase data
    (train-data
     (fann-internal:fann-train-on-data (%pointer nn) (%pointer data) 
				       max-epochs epochs-between-reports 
				       desired-error))
    ((or string pathname)
     (cffi:with-foreign-string (data-filename (namestring data))
       (fann-internal:fann-train-on-file (%pointer nn) data-filename
					 max-epochs epochs-between-reports
					 desired-error)))))

(defun train-epoch (nn data)
  "Train NN for a single epoch on DATA. Returns the MSE as calculated before or 
during training, rather than after training is complete."
  (fann-internal:fann-train-epoch (%pointer nn) (%pointer data)))

(defun test-on-data (nn data)
  "Test NN on DATA, updating and returning the MSE"
  (fann-internal:fann-test-data (%pointer nn) (%pointer data)))

(defun init-weights (nn data)
  (fann-internal:fann-init-weights (%pointer nn) (%pointer data)))

;;;; training parameters
(defun mse (nn)
  "Get the current MSE from training"
  (fann-internal:fann-get-mse (%pointer nn)))

(defun reset-mse (nn)
  "Reset the current MSE"
  (fann-internal:fann-reset-mse (%pointer nn)))