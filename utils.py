import numpy as np


def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.RNN_type,
        str(options.Ng),
        options.activation,
        'rf', str(options.place_cell_rf),
        'DoG', str(options.DoG),
        'periodic', str(options.periodic),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID


def get_2d_sort(x1,x2):
    """
    Reshapes x1 and x2 into square arrays, and then sorts
    them such that x1 increases downward and x2 increases
    rightward. Returns the order.
    """
    n = int(np.round(np.sqrt(len(x1))))
    total_order = x1.argsort()
    total_order = total_order.reshape(n,n)
    for i in range(n):
        row_order = x2[total_order.ravel()].reshape(n,n)[i].argsort()
        total_order[i] = total_order[i,row_order]
    total_order = total_order.ravel()
    return total_order


def load_trained_weights(model, trainer, weight_dir):
    ''' Load weights stored as a .npy file (for github)'''

    # Train for a single step to initialize weights
    print('Initialized trained weights.')
    trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array
    weights = np.load(weight_dir, allow_pickle=True)
    model.set_weights(weights)
    print('Loaded trained weights.')
    trainer.train(n_epochs=1, n_steps=1, save=False, evaluate=True)



