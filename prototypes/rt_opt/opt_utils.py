import numpy as np

f1_targ_rt = 1097
f2_targ_rt = 1495
f1_foil_rt = 1248
f2_foil_rt = 1727

f1_targ_err = 0.019
f2_targ_err = 0.08
f1_foil_err = 0.032
f2_foil_err = 0.09

f_in = ("fan1", "fan2", "foil1", "foil2")


def loss_func(errs, react_tms):
    total_loss = 0.0

    mean_rts = {ff: np.mean(react_tms[ff]) for ff in f_in}
    max_rt = np.max(list(mean_rts.values()))

    total_loss += np.abs(f1_targ_err - np.mean(errs["fan1"]))
    total_loss += np.abs(f2_targ_err - np.mean(errs["fan2"]))
    total_loss += np.abs(f1_foil_err - np.mean(errs["foil1"]))
    total_loss += np.abs(f2_foil_err - np.mean(errs["foil2"]))

    total_loss += np.abs(
        (f1_targ_rt - f2_targ_rt) / f2_foil_rt
        - (mean_rts["fan1"] - mean_rts["fan2"]) / max_rt)
    total_loss += np.abs(
        (f1_targ_rt - f1_foil_rt) / f2_foil_rt
        - (mean_rts["fan1"] - mean_rts["foil1"]) / max_rt)
    total_loss += np.abs(
        (f2_targ_rt - f2_foil_rt) / f2_foil_rt
        - (mean_rts["fan2"] - mean_rts["foil2"]) / max_rt)
    print(total_loss)

    return total_loss


def decision(match_output: np.ndarray, thresh=1.0, dec_thresh=0.5):
    below_thresh = match_output.copy()
    below_diff = thresh - below_thresh
    cum_below = np.cumsum(below_diff) / 100

    above_thresh = match_output.copy()
    above_diff = above_thresh - thresh
    cum_above = np.cumsum(above_diff) / 100

    above_cross = np.argmax(cum_above > dec_thresh)
    below_cross = np.argmax(cum_below > dec_thresh)
    b_crossed = (below_cross == 0 and cum_below[0] > dec_thresh) or below_cross > 0
    a_crossed = (above_cross == 0 and cum_above[0] > dec_thresh) or above_cross > 0

    dec_time = 0
    dec = 0

    if a_crossed and b_crossed:
        if above_cross < below_cross:
            dec_time = above_cross
            dec = 1
        elif below_cross < above_cross:
            dec_time = below_cross
            dec = -1
    elif a_crossed and not b_crossed:
        dec_time = above_cross
        dec = 1
    elif b_crossed and not a_crossed:
        dec_time = below_cross
        dec = -1
    elif not (below_cross or above_cross):
        dec_time = len(match_output)
        if cum_below[-1] > cum_above[-1]:
            dec = -1
        elif cum_above[-1] > cum_below[-1]:
            dec = 1
        else:
            dec = 0

    else:
        print("OMG WTF. IT WAS SUPPOSED TO BE TWO BOOLEANS.")
        dec_time = 0
        dec = 0

    return dec_time, dec
