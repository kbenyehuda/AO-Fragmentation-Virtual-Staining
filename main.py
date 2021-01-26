from virtual_staining import virtal_stain
from generator_function import build_5_generators
import matplotlib.pyplot as plt
import numpy as np

gens = build_5_generators()

virtually_stained_image = virtal_stain(1.0)
plt.imshow(np.squeeze(virtually_stained_image))

