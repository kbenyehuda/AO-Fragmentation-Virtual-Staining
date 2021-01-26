import numpy as np
from generator_function import build_5_generators

def cl_from_pred(pred):
  rnd = np.round(5*pred)
  if rnd == 0:
    return 1
  return int(rnd)

def virtal_stain(ao_frag_pred,gens):
    cl = cl_from_pred(ao_frag_pred)
    virtual_stain = gens[cl - 1].predict(np.random.normal(0, 1, (1, 100)))
    return virtual_stain

