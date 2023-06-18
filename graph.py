import pickle
import matplotlib.pyplot as plt
import numpy as np

dir = ['1.0_ll', '1.0_ll+sl', '1.0_ll_then_sl']
label = ['latent_loss', 'latent_loss+spatial_loss', 'latent_loss_then_spatial_loss']
epoch = range(199)
for c, i in enumerate(dir):
    file = open(f'history/ct/{i}.pkl', 'rb')
    history = pickle.load(file)
    plt.plot(epoch, history, label=label[c])

plt.legend(loc='lower left')
plt.show()
plt.savefig('1_history.png')