"""Random-circuit sampling on the IBM Nighthawk r2 processor (``ibm_phoenix``).

Library code behind the paper "Quantum computational advantage in random-circuit
sampling on a 120-qubit superconducting quantum computer":

``rcs.layout``       qubit layout, four-colouring of the square lattice, data loading
``rcs.placement``    calibration-aware choice of the 8x8 subgrid
``rcs.patching``     cut-minimising patch partitions and the pseudo-patched mirror filter
``rcs.circuits``     random-circuit, patched-circuit and mirror-circuit construction
``rcs.simulate``     exact ideal output probabilities of patch sub-circuits
``rcs.estimators``   patched linear XEB, mirror survival, pooling, decay fit, cost model
``rcs.io``           compact storage of measured bitstrings
``rcs.viz``          device, placement and partition figures
"""

__version__ = "1.0.0"
