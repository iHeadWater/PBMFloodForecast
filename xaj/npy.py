import numpy as np
import pandas as pd
from matplotlib._api import deprecated

import definitions


@deprecated
def test_read_npy():
    train_path = str(definitions.ROOT_DIR) + '/hydromodel/example/basins_lump_p_pe_q_fold0_train.npy'
    data = np.load(train_path)
    data = data.reshape(4017, 3)
    df = pd.DataFrame(data)
    df.to_csv('/home/wangjingyi/code/hydro-model-xaj/hydromodel/example/data.csv')