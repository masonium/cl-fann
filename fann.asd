(defpackage :cl-fann-system
  (:use :asdf))

(in-package :cl-fann-system)

(defsystem :fann
  :description "CFFI wrapper and lispy interface for libfann, an artificial neural network library"
  :author "Mason Smith <masonium@gmail.com>"
  :maintainer "Mason Smith <masonium@gmail.com>"
  :license "WTFPL"
  :depends-on (:cffi :trivial-garbage :alexandria)
  :components
  ((:file "fann-internal")
   (:file "package")
   (:file "util")
   (:file "fann")
   (:file "train"))
  :serial t)
