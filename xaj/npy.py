import numpy as np
import pandas as pd
import openpyxl
data = np.load('/home/wangjingyi/code/hydro-model-xaj/hydromodel/example/basins_lump_p_pe_q_fold0_train.npy')
data = data.reshape(4017, 3)
df = pd.DataFrame(data)
df.to_excel('/home/wangjingyi/code/hydro-model-xaj/hydromodel/example/data.xlsx') 