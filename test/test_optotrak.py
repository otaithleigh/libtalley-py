from pathlib import Path

import numpy as np

from libtalley.optotrak import load_optotrak

TESTROOT = Path(__file__).parent


def test_load_optotrak():
    data = load_optotrak(TESTROOT/'test_optotrak.tsv', delimiter='\t')
    assert np.allclose(data.loc[24, 'Column4'],
                       [72.362427, -320.086182, -2410.855713])
    assert data.attrs['Count'] == 500
    assert data.attrs['Frequency'] == 50.0
    assert data.attrs['Units'] == 'mm'
