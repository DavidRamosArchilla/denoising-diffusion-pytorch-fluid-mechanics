import numpy as np
from torch.utils.data import Dataset
import torch


class AirfoilDataset(Dataset):
    def __init__(self, pressures: list, coords: list, conditions: list, coefficients=None, max_len=None):
        """
        Args:
            pressures:   list of (N_i,)   arrays
            coords:      list of (N_i, 2) arrays  [x, y]
            conditions:  list of (2,)     arrays  [vel_inf, aoa]
        """
        # ------------------------------------------------------------------ #
        # 1. Compute statistics on the RAW data (before padding)             #
        # ------------------------------------------------------------------ #
        all_p    = np.concatenate(pressures)                    # (sum_Ni,)
        all_xy   = np.concatenate(coords, axis=0)               # (sum_Ni, 2)
        all_cond = np.stack(conditions, axis=0)                 # (M, 2)
        if coefficients is not None:
            self.p_mean, self.p_std = coefficients['p_mean'], coefficients['p_std']
            self.xy_mean, self.xy_std = coefficients['xy_mean'], coefficients['xy_std']
            self.c_mean, self.c_std = coefficients['c_mean'], coefficients['c_std']
        else:
            self.p_mean,  self.p_std  = all_p.mean(),   all_p.std()
            self.xy_mean, self.xy_std = all_xy.mean(0), all_xy.std(0)   # (2,) each
            self.c_mean,  self.c_std  = all_cond.mean(0), all_cond.std(0)

        # ------------------------------------------------------------------ #
        # 2. Standardize                                                      #
        # ------------------------------------------------------------------ #
        pressures_n  = [(p  - self.p_mean)  / self.p_std              for p  in pressures]
        coords_n     = [(xy - self.xy_mean) / self.xy_std             for xy in coords]
        conditions_n = [(c  - self.c_mean)  / self.c_std              for c  in conditions]

        # ------------------------------------------------------------------ #
        # 3. Pad to the longest sequence                                      #
        # ------------------------------------------------------------------ #
        # chapuzilla que solo funciona si si el maxlen de test seria menor que el de train, pero bueno, algun otro apaño se podria hacer
        if max_len is None:
            self.max_len = max(p.shape[0] for p in pressures_n)
        else:
            self.max_len = max_len
        M = len(pressures_n)

        p_pad  = np.zeros((M, self.max_len),    dtype=np.float32)
        xy_pad = np.zeros((M, self.max_len, 2), dtype=np.float32)
        mask   = np.zeros((M, self.max_len),    dtype=bool)          # False = padding

        for i, (p, xy) in enumerate(zip(pressures_n, coords_n)):
            n = p.shape[0]
            p_pad[i,  :n]    = p.astype(np.float32)
            xy_pad[i, :n]    = xy.astype(np.float32)
            mask[i,   :n]    = True                                  # real tokens

        self.pressures   = torch.from_numpy(p_pad).unsqueeze(1)      # (M, 1, L)
        self.coords      = torch.from_numpy(xy_pad).transpose(1, 2)  # (M, 2, L)
        self.conditions  = torch.from_numpy(
            np.array(conditions_n, dtype=np.float32))                # (M, 2)
        self.mask        = torch.from_numpy(mask)                    # (M, L)  bool
        print(f"Pressure mean/std: {self.pressures.mean():.4f}, {self.pressures.std():.4f}")
        print(f"Coords mean/std: {self.coords.mean((0, 2))}, {self.coords.std((0, 2))}")
        print(f"Conditions mean/std: {self.conditions.mean(0)}, {self.conditions.std(0)}")

    # ---------------------------------------------------------------------- #
    def __len__(self):
        return self.pressures.shape[0]

    def __getitem__(self, idx):
        """
        Returns
        -------
        pressure   : (L,)    - standardised, padded
        condition  : (2,)    - standardised [vel_inf, aoa]
        coords     : (L, 2)  - standardised [x, y], padded
        mask       : (L,)    - True for real points, False for padding
                               ready for scaled_dot_product_attention
        """
        return (
            self.pressures[idx],
            self.conditions[idx],
            self.coords[idx],
            self.mask[idx],
        )