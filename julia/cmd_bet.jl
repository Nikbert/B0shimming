
cmd = `bet image1_for_mask.nii bet_term.nii -m -n -f 0.65`

unz = `gunzip bet_term_mask.nii.gz`

run(cmd)

run(unz)
