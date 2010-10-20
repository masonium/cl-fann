(cl:in-package :cl-user)

(defpackage :cl-fann
  (:nicknames :fann)
  (:use :common-lisp :fann-internal :alexandria)
  (:documentation 
"CL-FANN is a CFFI wrapper and interface for the libfann artificial neural network library."))
