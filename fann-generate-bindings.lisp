(in-package :cl-user)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :asdf))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (asdf:operate 'asdf:load-op :verrazano)
  (asdf:operate 'asdf:load-op :cl-ppcre))

(in-package :verrazano-user)

(defun generate-binding* (name headers &rest args
                          &key 
                          (gccxml-flags "-I/usr/include")
                          &allow-other-keys)
  (format *debug-io* "~%~%; *** Processing binding ~S~%" name)
  (remove-from-plistf args :working-directory :gccxml-flags)
  (block try
    (handler-bind ((serious-condition
                    (lambda (error)
                      (warn "Failed to generated binding for ~S, error: ~A" name error)
                      (return-from try))))
      (let ((*print-right-margin* 100))
        (generate-binding (append
                           (list :cffi
                                 :package-name name
                                 :input-files headers
                                 :gccxml-flags gccxml-flags)
                           args)
                          :keep-temporary-files nil))))
  (values))

(defun generate-fann-binding ()
  (generate-binding*
   :fann-internal
   '("doublefann.h"
     "fann_train.h"
     "fann_data.h"
     "fann_cascade.h"
     "fann_activation.h"
     "fann_error.h"
     "fann_io.h")
   :package-nicknames "FANNINT"
   ;; drop "fann" prefix
   :standard-name-transformer-replacements
   `(,(cl-ppcre:create-scanner "(^FANN_?)") "")))
