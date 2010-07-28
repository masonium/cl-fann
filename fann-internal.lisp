;;; Generated by Verrazano 0.5
;;; WARNING: This is a generated file, editing it is unwise!


(cl:in-package :cl-user)

(cl:defpackage :fann-internal
  (:use :cffi)
  (:nicknames "FANNINT")
  (:export "FANN-LENGTH-TRAIN-DATA"
           "FANN-CREATE-SPARSE-ARRAY"
           "FANN-GET-RPROP-INCREASE-FACTOR"
           "FANN-DESTROY-TRAIN"
           "FANN-MERGE-TRAIN-DATA"
           "FANN-SAVE-TO-FIXED"
           "FANN-READ-TRAIN-FROM-FILE"
           "FANN-CREATE-FROM-FILE"
           "FANN-SET-CALLBACK"
           "FANN-PRINT-CONNECTIONS"
           "FANN-SET-LEARNING-RATE"
           "FANN-CREATE-STANDARD"
           "FANN-SET-CASCADE-ACTIVATION-STEEPNESSES"
           "FANN-SET-QUICKPROP-DECAY"
           "FANN-SET-TRAINING-ALGORITHM"
           "+TRAIN-NAMES+"
           "FANN-SET-CASCADE-WEIGHT-MULTIPLIER"
           "+STOPFUNC-NAMES+"
           "FANN-SET-ACTIVATION-STEEPNESS-OUTPUT"
           "+ACTIVATIONFUNC-NAMES+"
           "FANN-SET-ACTIVATION-FUNCTION-LAYER"
           "FANN-SHUFFLE-TRAIN-DATA"
           "FANN-SET-RPROP-INCREASE-FACTOR"
           "FANN-CREATE-SPARSE"
           "FANN-GET-QUICKPROP-MU"
           "FANN-TRAIN-EPOCH"
           "FANN-SET-ERROR-LOG"
           "FANN-GET-RPROP-DELTA-MAX"
           "FANN-GET-RPROP-DELTA-MIN"
           "FANN-SCALE-INPUT-TRAIN-DATA"
           "FANN-SET-ACTIVATION-FUNCTION-OUTPUT"
           "FANN-SAVE-TRAIN"
           "FANN-GET-NUM-OUTPUT"
           "FANN-RESET-ERRSTR"
           "FANN-GET-NUM-INPUT"
           "FANN-RESET-MSE"
           "FANN-SUBSET-TRAIN-DATA"
           "FANN-SET-CASCADE-OUTPUT-STAGNATION-EPOCHS"
           "FANN-GET-MSE"
           "FANN-TRAIN"
           "FANN-GET-CASCADE-ACTIVATION-FUNCTIONS"
           "FANN-SET-CASCADE-CANDIDATE-LIMIT"
           "FANN-PRINT-PARAMETERS"
           "FANN-SET-CASCADE-OUTPUT-CHANGE-FRACTION"
           "FANN-SAVE"
           "FANN-SCALE-OUTPUT-TRAIN-DATA"
           "FANN-NUM-OUTPUT-TRAIN-DATA"
           "FANN-TEST"
           "FANN-RESET-ERRNO"
           "FANN-GET-CASCADE-MAX-OUT-EPOCHS"
           "FANN-SET-CASCADE-NUM-CANDIDATE-GROUPS"
           "FANN-TEST-DATA"
           "FANN-SET-RPROP-DELTA-MIN"
           "FANN-GET-CASCADE-ACTIVATION-FUNCTIONS-COUNT"
           "FANN-SET-ACTIVATION-STEEPNESS-LAYER"
           "FANN-GET-TRAINING-ALGORITHM"
           "FANN-SET-CASCADE-CANDIDATE-STAGNATION-EPOCHS"
           "FANN-GET-ERRNO"
           "FANN-SET-RPROP-DELTA-MAX"
           "FANN-GET-ERRSTR"
           "FANN-GET-CASCADE-CANDIDATE-CHANGE-FRACTION"
           "FANN-SET-BIT-FAIL-LIMIT"
           "FANN-SET-ACTIVATION-STEEPNESS"
           "FANN-GET-CASCADE-NUM-CANDIDATES"
           "FANN-CREATE-STANDARD-ARRAY"
           "FANN-SET-RPROP-DECREASE-FACTOR"
           "FANN-PRINT-ERROR"
           "FANN-SCALE-TRAIN-DATA"
           "FANN-SET-TRAIN-STOP-FUNCTION"
           "FANN-ERROR"
           "FANN-NUM-INPUT-TRAIN-DATA"
           "+ERRORFUNC-NAMES+"
           "FANN-SET-CASCADE-MAX-OUT-EPOCHS"
           "FANN-SET-CASCADE-ACTIVATION-FUNCTIONS"
           "FANN-TRAIN-ON-DATA"
           "FANN-CASCADETRAIN-ON-DATA"
           "FANN-RUN"
           "FANN-CREATE-SHORTCUT"
           "FANN-GET-TOTAL-NEURONS"
           "FANN-GET-BIT-FAIL-LIMIT"
           "FANN-SAVE-TRAIN-TO-FIXED"
           "FANN-SET-CASCADE-CANDIDATE-CHANGE-FRACTION"
           "FANN-GET-TRAIN-STOP-FUNCTION"
           "FANN-GET-LEARNING-RATE"
           "FANN-RANDOMIZE-WEIGHTS"
           "FANN-GET-BIT-FAIL"
           "FANN-GET-QUICKPROP-DECAY"
           "FANN-SET-QUICKPROP-MU"
           "FANN-SET-CASCADE-MAX-CAND-EPOCHS"
           "FANN-SET-ACTIVATION-FUNCTION-HIDDEN"
           "FANN-GET-TOTAL-CONNECTIONS"
           "FANN-GET-RPROP-DECREASE-FACTOR"
           "FANN-TRAIN-ON-FILE"
           "FANN-DUPLICATE-TRAIN-DATA"
           "FANN-GET-CASCADE-OUTPUT-STAGNATION-EPOCHS"
           "FANN-GET-CASCADE-ACTIVATION-STEEPNESSES"
           "FANN-SET-ACTIVATION-FUNCTION"
           "FANN-GET-CASCADE-ACTIVATION-STEEPNESSES-COUNT"
           "FANN-INIT-WEIGHTS"
           "FANN-GET-CASCADE-NUM-CANDIDATE-GROUPS"
           "FANN-DEFAULT-ERROR-LOG"
           "FANN-GET-CASCADE-CANDIDATE-STAGNATION-EPOCHS"
           "INPUT"
           "NUM-DATA"
           "FANN-TRAIN-DATA"
           "FANN-GET-LEARNING-MOMENTUM"
           "FANN-SET-TRAIN-ERROR-FUNCTION"
           "FANN-CREATE-SHORTCUT-ARRAY"
           "FANN-GET-CASCADE-OUTPUT-CHANGE-FRACTION"
           "FANN-SET-LEARNING-MOMENTUM"
           "FANN-GET-CASCADE-MAX-CAND-EPOCHS"
           "FANN-DESTROY"
           "FANN-GET-TRAIN-ERROR-FUNCTION"
           "FANN-GET-CASCADE-CANDIDATE-LIMIT"
           "FANN-SET-ACTIVATION-STEEPNESS-HIDDEN"
           "FANN-CASCADETRAIN-ON-FILE"
           "FANN-GET-CASCADE-WEIGHT-MULTIPLIER"
           "PREV-WEIGHTS-DELTAS"
           "PREV-TRAIN-SLOPES"
           "PREV-STEPS"
           "TRAIN-SLOPES"
           "RPROP-DELTA-ZERO"
           "RPROP-DELTA-MAX"
           "RPROP-DELTA-MIN"
           "RPROP-DECREASE-FACTOR"
           "RPROP-INCREASE-FACTOR"
           "QUICKPROP-MU"
           "QUICKPROP-DECAY"
           "TOTAL-CONNECTIONS-ALLOCATED"
           "TOTAL-NEURONS-ALLOCATED"
           "CASCADE-CANDIDATE-SCORES"
           "CASCADE-NUM-CANDIDATE-GROUPS"
           "CASCADE-ACTIVATION-STEEPNESSES-COUNT"
           "CASCADE-ACTIVATION-STEEPNESSES"
           "CASCADE-ACTIVATION-FUNCTIONS-COUNT"
           "CASCADE-ACTIVATION-FUNCTIONS"
           "CASCADE-MAX-CAND-EPOCHS"
           "CASCADE-MAX-OUT-EPOCHS"
           "CASCADE-WEIGHT-MULTIPLIER"
           "CASCADE-CANDIDATE-LIMIT"
           "CASCADE-BEST-CANDIDATE"
           "CASCADE-CANDIDATE-STAGNATION-EPOCHS"
           "CASCADE-CANDIDATE-CHANGE-FRACTION"
           "CASCADE-OUTPUT-STAGNATION-EPOCHS"
           "CASCADE-OUTPUT-CHANGE-FRACTION"
           "CALLBACK"
           "TRAIN-STOP-FUNCTION"
           "TRAIN-ERROR-FUNCTION"
           "BIT-FAIL-LIMIT"
           "NUM-BIT-FAIL"
           "MSE-VALUE"
           "NUM-MSE"
           "OUTPUT"
           "TOTAL-CONNECTIONS"
           "TRAINING-ALGORITHM"
           "TRAIN-ERRORS"
           "CONNECTIONS"
           "WEIGHTS"
           "NUM-OUTPUT"
           "NUM-INPUT"
           "TOTAL-NEURONS"
           "LAST-LAYER"
           "FIRST-LAYER"
           "SHORTCUT-CONNECTIONS"
           "CONNECTION-RATE"
           "LEARNING-MOMENTUM"
           "LEARNING-RATE"
           "ERRSTR"
           "ERROR-LOG"
           "ERRNO-F"
           "FANN"
           "FANN-CALLBACK-TYPE"
           "STOPFUNC-BIT"
           "STOPFUNC-MSE"
           "FANN-STOPFUNC-ENUM"
           "ERRORFUNC-TANH"
           "ERRORFUNC-LINEAR"
           "FANN-ERRORFUNC-ENUM"
           "TRAIN-QUICKPROP"
           "TRAIN-RPROP"
           "TRAIN-BATCH"
           "TRAIN-INCREMENTAL"
           "FANN-TRAIN-ENUM"
           "LAST-NEURON"
           "FIRST-NEURON"
           "FANN-LAYER"
           "ACTIVATION-FUNCTION"
           "ACTIVATION-STEEPNESS"
           "VALUE"
           "SUM"
           "LAST-CON"
           "FIRST-CON"
           "FANN-NEURON"
           "LINEAR-PIECE-SYMMETRIC"
           "LINEAR-PIECE"
           "ELLIOT-SYMMETRIC"
           "ELLIOT"
           "GAUSSIAN-STEPWISE"
           "GAUSSIAN-SYMMETRIC"
           "GAUSSIAN"
           "SIGMOID-SYMMETRIC-STEPWISE"
           "SIGMOID-SYMMETRIC"
           "SIGMOID-STEPWISE"
           "SIGMOID"
           "THRESHOLD-SYMMETRIC"
           "THRESHOLD"
           "LINEAR"
           "FANN-ACTIVATIONFUNC-ENUM"
           "FILE"
           "SIZE-T"
           "E-INDEX-OUT-OF-BOUND"
           "E-TRAIN-DATA-SUBSET"
           "E-CANT-USE-TRAIN-ALG"
           "E-TRAIN-DATA-MISMATCH"
           "E-CANT-USE-ACTIVATION"
           "E-CANT-TRAIN-ACTIVATION"
           "E-CANT-ALLOCATE-MEM"
           "E-CANT-READ-TD"
           "E-CANT-OPEN-TD-R"
           "E-CANT-OPEN-TD-W"
           "E-WRONG-NUM-CONNECTIONS"
           "E-CANT-READ-CONNECTIONS"
           "E-CANT-READ-NEURON"
           "E-CANT-READ-CONFIG"
           "E-WRONG-CONFIG-VERSION"
           "E-CANT-OPEN-CONFIG-W"
           "E-CANT-OPEN-CONFIG-R"
           "E-NO-ERROR"
           "FANN-ERRNO-ENUM"
           "FANN-TYPE"))

(cl:in-package :fann-internal)

(define-foreign-library libfann
  (:unix "libfann.so")
  (:default "libfann"))

(cl:defun load-fann ()
  (cffi:use-foreign-library libfann))

(cl:defun close-fann ()
  (close-foreign-library 'libfann))

(cl:defun vtable-lookup (pobj indx coff)
  (cl:let ((vptr (cffi:mem-ref pobj :pointer coff)))
    (cffi:mem-aref vptr :pointer (cl:- indx 2))))

(cl:defmacro virtual-funcall (pobj indx coff cl:&body body)
  `(cffi:foreign-funcall-pointer (vtable-lookup ,pobj ,indx ,coff) ,cl:nil ,@body))

(cffi::defctype fann-type :double)

(cffi:defcenum fann-errno-enum
  (:e-no-error 0)
  (:e-cant-open-config-r 1)
  (:e-cant-open-config-w 2)
  (:e-wrong-config-version 3)
  (:e-cant-read-config 4)
  (:e-cant-read-neuron 5)
  (:e-cant-read-connections 6)
  (:e-wrong-num-connections 7)
  (:e-cant-open-td-w 8)
  (:e-cant-open-td-r 9)
  (:e-cant-read-td 10)
  (:e-cant-allocate-mem 11)
  (:e-cant-train-activation 12)
  (:e-cant-use-activation 13)
  (:e-train-data-mismatch 14)
  (:e-cant-use-train-alg 15)
  (:e-train-data-subset 16)
  (:e-index-out-of-bound 17))

(cffi:defcstruct _io-marker
  (_next :pointer)
  (_sbuf :pointer)
  (_pos :int))

(cffi::defctype _-off-t :long)

(cffi::defctype _io-lock-t :void)

(cffi::defctype _-quad-t :long-long)

(cffi::defctype _-off-64-t _-quad-t)

(cffi::defctype size-t :unsigned-int)

(cffi:defcstruct _io-file
  (_flags :int)
  (_io-read-ptr (:pointer :char))
  (_io-read-end (:pointer :char))
  (_io-read-base (:pointer :char))
  (_io-write-base (:pointer :char))
  (_io-write-ptr (:pointer :char))
  (_io-write-end (:pointer :char))
  (_io-buf-base (:pointer :char))
  (_io-buf-end (:pointer :char))
  (_io-save-base (:pointer :char))
  (_io-backup-base (:pointer :char))
  (_io-save-end (:pointer :char))
  (_markers :pointer)
  (_chain :pointer)
  (_fileno :int)
  (_flags-2 :int)
  (_old-offset _-off-t)
  (_cur-column :unsigned-short)
  (_vtable-offset :char)
  (_shortbuf :char :count 1)
  (_lock :pointer)
  (_offset _-off-64-t)
  (_-pad-1 (:pointer :void))
  (_-pad-2 (:pointer :void))
  (_-pad-3 (:pointer :void))
  (_-pad-4 (:pointer :void))
  (_-pad-5 size-t)
  (_mode :int)
  (_unused-2 :char :count 40))

(cffi::defctype file _io-file)

(cffi:defcenum fann-activationfunc-enum
  (:linear 0)
  (:threshold 1)
  (:threshold-symmetric 2)
  (:sigmoid 3)
  (:sigmoid-stepwise 4)
  (:sigmoid-symmetric 5)
  (:sigmoid-symmetric-stepwise 6)
  (:gaussian 7)
  (:gaussian-symmetric 8)
  (:gaussian-stepwise 9)
  (:elliot 10)
  (:elliot-symmetric 11)
  (:linear-piece 12)
  (:linear-piece-symmetric 13))

(cffi:defcstruct fann-neuron
  (first-con :unsigned-int)
  (last-con :unsigned-int)
  (sum fann-type)
  (value fann-type)
  (activation-steepness fann-type)
  (activation-function fann-activationfunc-enum))

(cffi:defcstruct fann-layer
  (first-neuron :pointer)
  (last-neuron :pointer))

(cffi:defcenum fann-train-enum
  (:train-incremental 0)
  (:train-batch 1)
  (:train-rprop 2)
  (:train-quickprop 3))

(cffi:defcenum fann-errorfunc-enum
  (:errorfunc-linear 0)
  (:errorfunc-tanh 1))

(cffi:defcenum fann-stopfunc-enum
  (:stopfunc-mse 0)
  (:stopfunc-bit 1))

(cffi::defctype fann-callback-type :pointer)

(cffi:defcstruct fann
  (errno-f fann-errno-enum)
  (error-log :pointer)
  (errstr (:pointer :char))
  (learning-rate :float)
  (learning-momentum :float)
  (connection-rate :float)
  (shortcut-connections :unsigned-int)
  (first-layer :pointer)
  (last-layer :pointer)
  (total-neurons :unsigned-int)
  (num-input :unsigned-int)
  (num-output :unsigned-int)
  (weights :pointer)
  (connections :pointer)
  (train-errors :pointer)
  (training-algorithm fann-train-enum)
  (total-connections :unsigned-int)
  (output :pointer)
  (num-mse :unsigned-int)
  (mse-value :float)
  (num-bit-fail :unsigned-int)
  (bit-fail-limit fann-type)
  (train-error-function fann-errorfunc-enum)
  (train-stop-function fann-stopfunc-enum)
  (callback fann-callback-type)
  (cascade-output-change-fraction :float)
  (cascade-output-stagnation-epochs :unsigned-int)
  (cascade-candidate-change-fraction :float)
  (cascade-candidate-stagnation-epochs :unsigned-int)
  (cascade-best-candidate :unsigned-int)
  (cascade-candidate-limit fann-type)
  (cascade-weight-multiplier fann-type)
  (cascade-max-out-epochs :unsigned-int)
  (cascade-max-cand-epochs :unsigned-int)
  (cascade-activation-functions :pointer)
  (cascade-activation-functions-count :unsigned-int)
  (cascade-activation-steepnesses :pointer)
  (cascade-activation-steepnesses-count :unsigned-int)
  (cascade-num-candidate-groups :unsigned-int)
  (cascade-candidate-scores :pointer)
  (total-neurons-allocated :unsigned-int)
  (total-connections-allocated :unsigned-int)
  (quickprop-decay :float)
  (quickprop-mu :float)
  (rprop-increase-factor :float)
  (rprop-decrease-factor :float)
  (rprop-delta-min :float)
  (rprop-delta-max :float)
  (rprop-delta-zero :float)
  (train-slopes :pointer)
  (prev-steps :pointer)
  (prev-train-slopes :pointer)
  (prev-weights-deltas :pointer))

(cffi:defcfun ("fann_get_cascade_weight_multiplier" fann-get-cascade-weight-multiplier) fann-type
                                                                                                  (ann :pointer))

(cffi:defcfun ("fann_cascadetrain_on_file" fann-cascadetrain-on-file) :void (ann :pointer)
                                                                            (filename :pointer)
                                                                            (max-neurons :unsigned-int)
                                                                            (neurons-between-reports :unsigned-int)
                                                                            (desired-error :float))

(cffi:defcfun ("fann_set_activation_steepness_hidden" fann-set-activation-steepness-hidden) :void
                                                                                                  (ann :pointer)
                                                                                                  (steepness fann-type))

(cffi:defcfun ("fann_get_cascade_candidate_limit" fann-get-cascade-candidate-limit) fann-type
                                                                                              (ann :pointer))

(cffi:defcfun ("fann_get_train_error_function" fann-get-train-error-function) fann-errorfunc-enum
                                                                                                  (ann :pointer))

(cffi:defcfun ("fann_destroy" fann-destroy) :void (ann :pointer))

(cffi:defcfun ("fann_get_cascade_max_cand_epochs" fann-get-cascade-max-cand-epochs) :unsigned-int
                                                                                                  (ann :pointer))

(cffi:defcfun ("fann_set_learning_momentum" fann-set-learning-momentum) :void (ann :pointer)
                                                                              (learning-momentum :float))

(cffi:defcfun ("fann_get_cascade_output_change_fraction" fann-get-cascade-output-change-fraction) :float
                                                                                                         (ann :pointer))

(cffi:defcfun ("fann_create_shortcut_array" fann-create-shortcut-array) :pointer
                                                                                 (num-layers :unsigned-int)
                                                                                 (layers (:pointer :unsigned-int)))

(cffi:defcfun ("fann_set_train_error_function" fann-set-train-error-function) :void (ann :pointer)
                                                                                    (train-error-function fann-errorfunc-enum))

(cffi:defcfun ("fann_get_learning_momentum" fann-get-learning-momentum) :float (ann :pointer))

(cffi:defcstruct fann-train-data
  (errno-f fann-errno-enum)
  (error-log :pointer)
  (errstr (:pointer :char))
  (num-data :unsigned-int)
  (num-input :unsigned-int)
  (num-output :unsigned-int)
  (input :pointer)
  (output :pointer))

(cffi:defcfun ("fann_get_cascade_candidate_stagnation_epochs" fann-get-cascade-candidate-stagnation-epochs) :unsigned-int
                                                                                                                          (ann :pointer))

(cffi:defcvar ("fann_default_error_log" fann-default-error-log) :pointer)

(cffi:defcfun ("fann_get_cascade_num_candidate_groups" fann-get-cascade-num-candidate-groups) :unsigned-int
                                                                                                            (ann :pointer))

(cffi:defcfun ("fann_init_weights" fann-init-weights) :void (ann :pointer) (train-data :pointer))

(cffi:defcfun ("fann_get_cascade_activation_steepnesses_count" fann-get-cascade-activation-steepnesses-count) :unsigned-int
                                                                                                                            (ann :pointer))

(cffi:defcfun ("fann_set_activation_function" fann-set-activation-function) :void (ann :pointer)
                                                                                  (activation-function fann-activationfunc-enum)
                                                                                  (layer :int)
                                                                                  (neuron :int))

(cffi:defcfun ("fann_get_cascade_activation_steepnesses" fann-get-cascade-activation-steepnesses) :pointer
                                                                                                           (ann :pointer))

(cffi:defcfun ("fann_get_cascade_output_stagnation_epochs" fann-get-cascade-output-stagnation-epochs) :unsigned-int
                                                                                                                    (ann :pointer))

(cffi:defcfun ("fann_duplicate_train_data" fann-duplicate-train-data) :pointer (data :pointer))

(cffi:defcfun ("fann_train_on_file" fann-train-on-file) :void (ann :pointer) (filename :pointer)
                                                              (max-epochs :unsigned-int)
                                                              (epochs-between-reports :unsigned-int)
                                                              (desired-error :float))

(cffi:defcfun ("fann_get_rprop_decrease_factor" fann-get-rprop-decrease-factor) :float
                                                                                       (ann :pointer))

(cffi:defcfun ("fann_get_total_connections" fann-get-total-connections) :unsigned-int (ann :pointer))

(cffi:defcfun ("fann_set_activation_function_hidden" fann-set-activation-function-hidden) :void
                                                                                                (ann :pointer)
                                                                                                (activation-function fann-activationfunc-enum))

(cffi:defcfun ("fann_set_cascade_max_cand_epochs" fann-set-cascade-max-cand-epochs) :void
                                                                                          (ann :pointer)
                                                                                          (cascade-max-cand-epochs :unsigned-int))

(cffi:defcfun ("fann_set_quickprop_mu" fann-set-quickprop-mu) :void (ann :pointer)
                                                                    (quickprop-mu :float))

(cffi:defcfun ("fann_get_quickprop_decay" fann-get-quickprop-decay) :float (ann :pointer))

(cffi:defcfun ("fann_get_bit_fail" fann-get-bit-fail) :unsigned-int (ann :pointer))

(cffi:defcfun ("fann_randomize_weights" fann-randomize-weights) :void (ann :pointer)
                                                                      (min-weight fann-type)
                                                                      (max-weight fann-type))

(cffi:defcfun ("fann_get_learning_rate" fann-get-learning-rate) :float (ann :pointer))

(cffi:defcfun ("fann_get_train_stop_function" fann-get-train-stop-function) fann-stopfunc-enum
                                                                                               (ann :pointer))

(cffi:defcfun ("fann_set_cascade_candidate_change_fraction" fann-set-cascade-candidate-change-fraction) :void
                                                                                                              (ann :pointer)
                                                                                                              (cascade-candidate-change-fraction :float))

(cffi:defcfun ("fann_save_train_to_fixed" fann-save-train-to-fixed) :int (data :pointer)
                                                                         (filename :pointer)
                                                                         (decimal-point :unsigned-int))

(cffi:defcfun ("fann_get_bit_fail_limit" fann-get-bit-fail-limit) fann-type (ann :pointer))

(cffi:defcfun ("fann_get_total_neurons" fann-get-total-neurons) :unsigned-int (ann :pointer))

(cffi:defcfun ("fann_create_shortcut" fann-create-shortcut) :pointer
                                                                     (num-layers :unsigned-int)common-lisp:&rest)

(cffi:defcfun ("fann_run" fann-run) :pointer (ann :pointer) (input :pointer))

(cffi:defcfun ("fann_cascadetrain_on_data" fann-cascadetrain-on-data) :void (ann :pointer)
                                                                            (data :pointer)
                                                                            (max-neurons :unsigned-int)
                                                                            (neurons-between-reports :unsigned-int)
                                                                            (desired-error :float))

(cffi:defcfun ("fann_train_on_data" fann-train-on-data) :void (ann :pointer) (data :pointer)
                                                              (max-epochs :unsigned-int)
                                                              (epochs-between-reports :unsigned-int)
                                                              (desired-error :float))

(cffi:defcfun ("fann_set_cascade_activation_functions" fann-set-cascade-activation-functions) :void
                                                                                                    (ann :pointer)
                                                                                                    (cascade-activation-functions :pointer)
                                                                                                    (cascade-activation-functions-count :unsigned-int))

(cffi:defcfun ("fann_set_cascade_max_out_epochs" fann-set-cascade-max-out-epochs) :void
                                                                                        (ann :pointer)
                                                                                        (cascade-max-out-epochs :unsigned-int))

(cl:defconstant +errorfunc-names+ cl:nil)

(cffi:defcfun ("fann_num_input_train_data" fann-num-input-train-data) :unsigned-int (data :pointer))

(cffi:defcstruct fann-error
  (errno-f fann-errno-enum)
  (error-log :pointer)
  (errstr (:pointer :char)))

(cffi:defcfun ("fann_set_train_stop_function" fann-set-train-stop-function) :void (ann :pointer)
                                                                                  (train-stop-function fann-stopfunc-enum))

(cffi:defcfun ("fann_scale_train_data" fann-scale-train-data) :void (train-data :pointer)
                                                                    (new-min fann-type)
                                                                    (new-max fann-type))

(cffi:defcfun ("fann_print_error" fann-print-error) :void (errdat :pointer))

(cffi:defcfun ("fann_set_rprop_decrease_factor" fann-set-rprop-decrease-factor) :void (ann :pointer)
                                                                                      (rprop-decrease-factor :float))

(cffi:defcfun ("fann_create_standard_array" fann-create-standard-array) :pointer
                                                                                 (num-layers :unsigned-int)
                                                                                 (layers (:pointer :unsigned-int)))

(cffi:defcfun ("fann_get_cascade_num_candidates" fann-get-cascade-num-candidates) :unsigned-int
                                                                                                (ann :pointer))

(cffi:defcfun ("fann_set_activation_steepness" fann-set-activation-steepness) :void (ann :pointer)
                                                                                    (steepness fann-type)
                                                                                    (layer :int)
                                                                                    (neuron :int))

(cffi:defcfun ("fann_set_bit_fail_limit" fann-set-bit-fail-limit) :void (ann :pointer)
                                                                        (bit-fail-limit fann-type))

(cffi:defcfun ("fann_get_cascade_candidate_change_fraction" fann-get-cascade-candidate-change-fraction) :float
                                                                                                               (ann :pointer))

(cffi:defcfun ("fann_get_errstr" fann-get-errstr) (:pointer :char) (errdat :pointer))

(cffi:defcfun ("fann_set_rprop_delta_max" fann-set-rprop-delta-max) :void (ann :pointer)
                                                                          (rprop-delta-max :float))

(cffi:defcfun ("fann_get_errno" fann-get-errno) fann-errno-enum (errdat :pointer))

(cffi:defcfun ("fann_set_cascade_candidate_stagnation_epochs" fann-set-cascade-candidate-stagnation-epochs) :void
                                                                                                                  (ann :pointer)
                                                                                                                  (cascade-candidate-stagnation-epochs :unsigned-int))

(cffi:defcfun ("fann_get_training_algorithm" fann-get-training-algorithm) fann-train-enum
                                                                                          (ann :pointer))

(cffi:defcfun ("fann_set_activation_steepness_layer" fann-set-activation-steepness-layer) :void
                                                                                                (ann :pointer)
                                                                                                (steepness fann-type)
                                                                                                (layer :int))

(cffi:defcfun ("fann_get_cascade_activation_functions_count" fann-get-cascade-activation-functions-count) :unsigned-int
                                                                                                                        (ann :pointer))

(cffi:defcfun ("fann_set_rprop_delta_min" fann-set-rprop-delta-min) :void (ann :pointer)
                                                                          (rprop-delta-min :float))

(cffi:defcfun ("fann_test_data" fann-test-data) :float (ann :pointer) (data :pointer))

(cffi:defcfun ("fann_set_cascade_num_candidate_groups" fann-set-cascade-num-candidate-groups) :void
                                                                                                    (ann :pointer)
                                                                                                    (cascade-num-candidate-groups :unsigned-int))

(cffi:defcfun ("fann_get_cascade_max_out_epochs" fann-get-cascade-max-out-epochs) :unsigned-int
                                                                                                (ann :pointer))

(cffi:defcfun ("fann_reset_errno" fann-reset-errno) :void (errdat :pointer))

(cffi:defcfun ("fann_test" fann-test) :pointer (ann :pointer) (input :pointer)
                                               (desired-output :pointer))

(cffi:defcfun ("fann_num_output_train_data" fann-num-output-train-data) :unsigned-int
                                                                                      (data :pointer))

(cffi:defcfun ("fann_scale_output_train_data" fann-scale-output-train-data) :void
                                                                                  (train-data :pointer)
                                                                                  (new-min fann-type)
                                                                                  (new-max fann-type))

(cffi:defcfun ("fann_save" fann-save) :int (ann :pointer) (configuration-file :pointer))

(cffi:defcfun ("fann_set_cascade_output_change_fraction" fann-set-cascade-output-change-fraction) :void
                                                                                                        (ann :pointer)
                                                                                                        (cascade-output-change-fraction :float))

(cffi:defcfun ("fann_print_parameters" fann-print-parameters) :void (ann :pointer))

(cffi:defcfun ("fann_set_cascade_candidate_limit" fann-set-cascade-candidate-limit) :void
                                                                                          (ann :pointer)
                                                                                          (cascade-candidate-limit fann-type))

(cffi:defcfun ("fann_get_cascade_activation_functions" fann-get-cascade-activation-functions) :pointer
                                                                                                       (ann :pointer))

(cffi:defcfun ("fann_train" fann-train) :void (ann :pointer) (input :pointer)
                                              (desired-output :pointer))

(cffi:defcfun ("fann_get_MSE" fann-get-mse) :float (ann :pointer))

(cffi:defcfun ("fann_set_cascade_output_stagnation_epochs" fann-set-cascade-output-stagnation-epochs) :void
                                                                                                            (ann :pointer)
                                                                                                            (cascade-output-stagnation-epochs :unsigned-int))

(cffi:defcfun ("fann_subset_train_data" fann-subset-train-data) :pointer (data :pointer)
                                                                         (pos :unsigned-int)
                                                                         (length :unsigned-int))

(cffi:defcfun ("fann_reset_MSE" fann-reset-mse) :void (ann :pointer))

(cffi:defcfun ("fann_get_num_input" fann-get-num-input) :unsigned-int (ann :pointer))

(cffi:defcfun ("fann_reset_errstr" fann-reset-errstr) :void (errdat :pointer))

(cffi:defcfun ("fann_get_num_output" fann-get-num-output) :unsigned-int (ann :pointer))

(cffi:defcfun ("fann_save_train" fann-save-train) :int (data :pointer) (filename :pointer))

(cffi:defcfun ("fann_set_activation_function_output" fann-set-activation-function-output) :void
                                                                                                (ann :pointer)
                                                                                                (activation-function fann-activationfunc-enum))

(cffi:defcfun ("fann_scale_input_train_data" fann-scale-input-train-data) :void
                                                                                (train-data :pointer)
                                                                                (new-min fann-type)
                                                                                (new-max fann-type))

(cffi:defcfun ("fann_get_rprop_delta_min" fann-get-rprop-delta-min) :float (ann :pointer))

(cffi:defcfun ("fann_get_rprop_delta_max" fann-get-rprop-delta-max) :float (ann :pointer))

(cffi:defcfun ("fann_set_error_log" fann-set-error-log) :void (errdat :pointer) (log-file :pointer))

(cffi:defcfun ("fann_train_epoch" fann-train-epoch) :float (ann :pointer) (data :pointer))

(cffi:defcfun ("fann_get_quickprop_mu" fann-get-quickprop-mu) :float (ann :pointer))

(cffi:defcfun ("fann_create_sparse" fann-create-sparse) :pointer (connection-rate :float)
                                                                 (num-layers :unsigned-int)common-lisp:&rest)

(cffi:defcfun ("fann_set_rprop_increase_factor" fann-set-rprop-increase-factor) :void (ann :pointer)
                                                                                      (rprop-increase-factor :float))

(cffi:defcfun ("fann_shuffle_train_data" fann-shuffle-train-data) :void (train-data :pointer))

(cffi:defcfun ("fann_set_activation_function_layer" fann-set-activation-function-layer) :void
                                                                                              (ann :pointer)
                                                                                              (activation-function fann-activationfunc-enum)
                                                                                              (layer :int))

(cl:defconstant +activationfunc-names+ cl:nil)

(cffi:defcfun ("fann_set_activation_steepness_output" fann-set-activation-steepness-output) :void
                                                                                                  (ann :pointer)
                                                                                                  (steepness fann-type))

(cl:defconstant +stopfunc-names+ cl:nil)

(cffi:defcfun ("fann_set_cascade_weight_multiplier" fann-set-cascade-weight-multiplier) :void
                                                                                              (ann :pointer)
                                                                                              (cascade-weight-multiplier fann-type))

(cl:defconstant +train-names+ cl:nil)

(cffi:defcfun ("fann_set_training_algorithm" fann-set-training-algorithm) :void (ann :pointer)
                                                                                (training-algorithm fann-train-enum))

(cffi:defcfun ("fann_set_quickprop_decay" fann-set-quickprop-decay) :void (ann :pointer)
                                                                          (quickprop-decay :float))

(cffi:defcfun ("fann_set_cascade_activation_steepnesses" fann-set-cascade-activation-steepnesses) :void
                                                                                                        (ann :pointer)
                                                                                                        (cascade-activation-steepnesses :pointer)
                                                                                                        (cascade-activation-steepnesses-count :unsigned-int))

(cffi:defcfun ("fann_create_standard" fann-create-standard) :pointer
                                                                     (num-layers :unsigned-int)common-lisp:&rest)

(cffi:defcfun ("fann_set_learning_rate" fann-set-learning-rate) :void (ann :pointer)
                                                                      (learning-rate :float))

(cffi:defcfun ("fann_print_connections" fann-print-connections) :void (ann :pointer))

(cffi:defcfun ("fann_set_callback" fann-set-callback) :void (ann :pointer)
                                                            (callback fann-callback-type))

(cffi:defcfun ("fann_create_from_file" fann-create-from-file) :pointer (configuration-file :pointer))

(cffi:defcfun ("fann_read_train_from_file" fann-read-train-from-file) :pointer (filename :pointer))

(cffi:defcfun ("fann_save_to_fixed" fann-save-to-fixed) :int (ann :pointer)
                                                             (configuration-file :pointer))

(cffi:defcfun ("fann_merge_train_data" fann-merge-train-data) :pointer (data-1 :pointer)
                                                                       (data-2 :pointer))

(cffi:defcfun ("fann_destroy_train" fann-destroy-train) :void (train-data :pointer))

(cffi:defcfun ("fann_get_rprop_increase_factor" fann-get-rprop-increase-factor) :float
                                                                                       (ann :pointer))

(cffi:defcfun ("fann_create_sparse_array" fann-create-sparse-array) :pointer
                                                                             (connection-rate :float)
                                                                             (num-layers :unsigned-int)
                                                                             (layers (:pointer :unsigned-int)))

(cffi:defcfun ("fann_length_train_data" fann-length-train-data) :unsigned-int (data :pointer))
