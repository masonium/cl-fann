;;;; util.lisp
;;;; Various utility functions
(in-package :fann)

(defun %map-nth-list (fn n list &rest more-lists)
  "Similar to MAPLIST, but uses every N'th CDR, rather than
CDR. (map-nth-list #'identity 2 '(1 2 3 4 5)) => '((1 2 3 4 5) (3 4
5) (5)))"
  (labels ((aux (acc list rest)
	     (if (null list)
		 (nreverse acc)
		 (aux
		  (cons (apply fn list rest) acc)
		  (nthcdr n list)
		  (mapcar #'(lambda (x) (nthcdr n x)) rest)))))
    (aux nil list more-lists)))
	   