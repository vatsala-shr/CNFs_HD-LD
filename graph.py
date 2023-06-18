import pickle
import matplotlib.pyplot as plt
import numpy as np

# dir = ['0.0_ll', '0.0_ll+sl', '0.0_ll_then_sl']
dir = ['0.0_ll', '0.0_ll_bs-2']
# label = ['latent_loss', 'latent_loss+spatial_loss', 'latent_loss_then_spatial_loss']
label = ['batch_size-1', 'batch_size-2']
epoch = range(199)
for c, i in enumerate(dir):
    file = open(f'history/ct/{i}.pkl', 'rb')
    history = pickle.load(file)
    plt.plot(epoch, history, label=label[c])

plt.legend(loc='lower left')
plt.show()
plt.savefig('0_bs_history.png')