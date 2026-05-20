import numpy as np
def compute_woe_iv(bins):
    bins['Dst_Gds'] = bins['goods'] / (bins['goods'] + bins['bads'])
    bins['Dst_Bds'] = bins['bads'] / (bins['goods'] + bins['bads'])
    bins['WOE'] = np.log(bins['goods'] / bins['bads'])
    bins['IV_component'] = (bins['Dst_Gds'] - bins['Dst_Bds']) * bins['WOE']
    IV = bins['IV_component'].sum(skipna=True)
    return bins, IV