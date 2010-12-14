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
  (with-slots (num-inputs num-outputs length) object
    (print-unreadable-object (object stream :type t :identity t)
      (format stream "~d, ~d, ~d" num-inputs num-outputs length)))
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

;;;; Accessors
(defun num-inputs-train-data (data)
  (slot-value data 'num-inputs))

(defun num-outputs-train-data (data)
  (slot-value data 'num-outputs))

(defun length-train-data (data)
  (slot-value data 'length))

;;;; Modifiers
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
    (%make-train-data (fann-internal:fann-subset-train-data (%pointer data) pos len))))

(defun read-train-data-from-file (pathname)
  "Create a training set with data loaded from PATHNAME"
  (cffi:with-foreign-string (data-filename (namestring pathname))
    (%make-train-data 
     (fann-read-train-from-file data-filename))))

(defun format-train-data (stream inputs outputs 
			  &key (input-fn #'identity) (output-fn #'identity))
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
      (format stream "~d ~d ~d~%" N 
	      (length (convert-data (funcall input-fn (elt inputs 0))))
	      (length (convert-data (funcall output-fn (elt outputs 0)))))
      (map nil
	   #'(lambda (in out)
	       (format stream "~{~5$~^ ~}~%~{~5$~^ ~}~%" 
		       (convert-data (funcall input-fn in)) 
		       (convert-data (funcall output-fn out))))
	   inputs
	   outputs))))

(defun scale-train-data (train-data new-min new-max 
			 &optional (input t) (output t))
  "Scale the inputs and/or outputs in TRAIN-DATA to the range (NEW-MIN, NEW-MAX)"
  (let ((pointer (%pointer train-data)))
    (funcall 
     (cond 
       ((and input output) #'fann-scale-train-data)
       (input #'fann-scale-input-train-data)
       (output #'fann-scale-output-train-data)
       (t #'(lambda (x y)
	      (declare (ignore x y)) nil)))
     pointer (coerce  new-min 'double-float) (coerce new-max 'double-float))))

(defun shuffle-train-data (train-data)
  "Shuffle the training data to be in a random order"
  (fann-shuffle-train-data (%pointer train-data)))

(defun train (nn input desired-output)
  "Train NN on a single (INPUT, DESIRED-OUTPUT) data pair"
  (check-raw-dimensions nn (length input) (length desired-output))
  (with-sequence-as-foreign-array (in input 'fann-internal:fann-type 
				      out desired-output 'fann-internal:fann-type)
    (fann-train (%pointer nn) in out)))

(defun test (nn input desired-output)
  "Test NN on a single data pair, updating the internal mse"
  (check-raw-dimensions nn (length input) (length desired-output))
  (with-sequence-as-foreign-array (in input 'fann-internal:fann-type
				      out desired-output 'fann-internal:fann-type)
    (fann-test (%pointer nn) in out)))

(defun train-on-data (nn data max-epochs epochs-between-reports desired-error)
  "Train NN on the DATA. DATA can be either a PATHNAME to a file in
the correct data format or a pre-loaded TRAIN-DATA dataset"
  (etypecase data
    (train-data
     (check-data-dimensions nn data)
     (fann-internal:fann-train-on-data (%pointer nn) (%pointer data) 
				       max-epochs epochs-between-reports 
				       desired-error))
    ((or string pathname)
     (cffi:with-foreign-string (data-filename (namestring data))
       (fann-internal:fann-train-on-file (%pointer nn) data-filename
					 max-epochs epochs-between-reports
					 desired-error)))))

(defun train-epoch (nn data)
  "Train NN for a single epoch on DATA. Returns the MSE as calculated
before or during training, rather than after training is
complete. DATA can be a pathname or a TRAIN-DATA object."
  (let ((loaded-dat (ensure-loaded-data data))))
  (fann-internal:fann-train-epoch (%pointer nn) (%pointer data)))

(defun test-on-data (nn data)
  "Test NN on DATA, updating and returning the MSE. DATA can either be
a pathname or an already-loaded TRAIN-DATA object."
  (let ((loaded-data (ensure-loaded-data data)))
    (check-data-dimensions nn loaded-data)
    (fann-internal:fann-test-data (%pointer nn) (%pointer loaded-data))))

(defun ensure-loaded-data (data)
  "If DATA is a TRAIN-DATA object, this function just returns it. If
  it is a string or pathname, it loads the data from file into a
  TRAIN-DATA object and returns it."
  (etypecase data
    (train-data data)
    ((or string pathname)
     (read-train-data-from-file data))))

(defun init-weights (nn data)
  (fann-internal:fann-init-weights (%pointer nn) (%pointer data)))

(defun check-data-dimensions (nn data)
  (when (not (= (num-input nn) (num-inputs-train-data data)))
    (error "Neural network and training data do not have same input dimension (~d vs. ~d)"
	   (num-input nn) (num-inputs-train-data data)))
  (when (not (= (num-output nn) (num-outputs-train-data data)))
    (error "Neural network and training data do not have same output dimension (~d vs. ~d"
	   (num-output nn) (num-outputs-train-data data))))

(defun check-raw-dimensions (nn in out)
  (when (not (= (num-input nn) in))
    (error "Neural network and datum do not have same input dimension (~d vs. ~d)"
	   (num-input nn) in))
  (when (not (= (num-output nn) out))
    (error "Neural network and datum do not have same output dimension (~d vs. ~d)"
	   (num-output nn) out)))

;;;; training parameters
(defun mse (nn)
  "Get the current MSE from training"
  (fann-internal:fann-get-mse (%pointer nn)))

(defun reset-mse (nn)
  "Reset the current MSE"
  (fann-internal:fann-reset-mse (%pointer nn)))
