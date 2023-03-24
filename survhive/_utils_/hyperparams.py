from .scorer import *

CVSCORERFACTORY = {
    "linear_predictor": linear_cv,
    "regular": basic_cv_fold,
    "vvh": vvh_cv_fold,
}
