import os
import numpy as np
import matplotlib.pyplot as plt

from ..data.ssm import lin_gaussian_ssm_loader as data_loader
from ..data.ssm import plot_particle_ssm_1d, plot_lin_gaussian_ssm_1d
from ..data.ssm import lin_gaussian_ssm_log_likelihood as log_likelihood

from .utils import get_default_callbacks

def main(params, model, repr='gaussian'):

    # data
    data_trn_target_ssm_state, data_trn_target_ssm_cov, data_trn_obs = data_loader(params, mode='trn')
    data_val_target_ssm_state, data_val_target_ssm_cov, data_val_obs = data_loader(params, mode='val')
    data_tst_target_ssm_state, data_tst_target_ssm_cov, data_tst_obs = data_loader(params, mode='tst')

    # train
    model.fit(
        data_trn_obs,
        batch_size=params['BATCH_SIZE'],
        epochs=params['NUM_EPOCHS'],
        validation_data=(data_val_obs,),
        callbacks=get_default_callbacks(params)
        )

    # evaluate
    outputs = model(data_tst_obs)
    outputs = [o.numpy() for o in outputs]

    # likelihood
    log_llh = log_likelihood(params, data_tst_target_ssm_state, data_tst_obs)
    print(np.mean(log_llh))

    log_llh = model.log_likelihood(data_tst_obs).numpy()
    print(np.mean(log_llh))

    # plot
    fig, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(20, 20))
    for r in range(4):
        for c in range(4):
            axs[r, c].set_ylim(ymin=data_tst_obs.min()*1.5, ymax=data_tst_obs.max()*1.5)
            plot_lin_gaussian_ssm_1d(
                axs[r, c],
                params,
                data_tst_target_ssm_state[r*c],
                np.squeeze(data_tst_target_ssm_cov[r*c]),
                data_tst_obs[r*c],
                c='k'
            )

            if repr == 'gaussian':
                plot_fn = plot_lin_gaussian_ssm_1d
            elif repr == 'particle':
                plot_fn = plot_particle_ssm_1d
            plot_fn(
                    axs[r, c],
                    params,
                    outputs[0][r*c],
                    outputs[1][r*c],
                    c='b'
                )

    plt.savefig(os.path.join(params['MODEL_PATH'], 'result'), format='jpg', dpi=300)
    plt.show()