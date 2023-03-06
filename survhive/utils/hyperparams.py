from scorer import *

CVSCORERFACTORY = {"linear_predictor": linear_cv, "regular": basic_cv, "vvh": vvh_cv}
ESTIMATORFACTORY = {}
OPTIMISERFACTORY = {}
