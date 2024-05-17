using MRIFieldmaps: b0map, b0model, b0init, b0scale, phasecontrast
using MIRTjim: jim, prompt; jim(:prompt, false)
using Statistics: mean
using Unitful: Hz, s
using Plots; default(markerstrokecolor=:auto, label="")
using ROMEO: unwrap
using NIfTI: niread
using MAT, JLD2
using MIRT: embed!

include("Shimming/getSHbasis.jl")     # getSHbasis()
include("Shimming/getcalmatrix.jl")   # getcalmatrix()
include("Shimming/shimoptim.jl")      # shimoptim()
include("phant_mask.jl")     # phant_mask()

clim = (-60, 60).*Hz
# Open Recontructed Data
data = matopen("nov10_1.mat")
img1r = read(data, "im_te1")
img2r = read(data, "im_te2")
echo = vec(read(data, "echotimes"))

mask1 = niread("image1_for_mask_brain65_mask.nii")
mask1 = mask1.raw
mask1 = iszero.(iszero.(mask1))

# Set Needed Parameters
ncoil = 32              # Number of coils
necho = length(echo)    # Number of Echos
l2b = -26               # Regularization parameter
precon = :diag;         # Precondition
echo = echo * 1f-3      # Echo time in seconds
l = 2                   # 1 for linear shims, 2 for second order
TH = 0.4                # Default threshold for mask
mask_flag = 0           # Initalization of mask flag
fov = [24 24 24]        # Field of view in cm

img1 = sqrt.(sum(abs.(img1r),dims = 4));
mask = []
mask_temp = []
dd = jim(mask1; title = "Input Mask")
display(dd)

println("Need thresholding? [y/n]")
thr = readline();

if thr == "y"
    while mask_flag == 0
        global TH, mask_flag, mask, mask_temp
        mask = phant_mask(img1,TH)
        mask_temp = mask1.*mask[:,:,:,1]
        pl = jim(mask_temp, title = "Mask")
        display(pl)
        println("Is mask sufficient? [y/n]")
        mask_good = readline();
        if mask_good == "y"
            mask_flag =1
        else
            println("Enter threshold ratio? [0:1]")
            println("Current Threshold:", TH)
            TH = parse(Float32,readline());
        end
    end
    mask = mask_temp
else
    mask = mask1
end

# Combine and scale complex data
images = cat(img1r, img2r; dims = 5)
img_sc = sum(images; dims=4)
(img_scale, scale) = b0scale(img_sc, echo)
images = images/scale;

# B0 Iniitalization and Unwrap
init1 = b0init(images,echo;)                    # Returns Inital Map in Hz
init = init1.*mask  
init_phase = (echo[2]-echo[1])*init*(2*pi)      # Convert to Radians
uw_init1 = unwrap(init_phase[:,:,:,1]; mag = img1, mask = mask)  # Unwrap Returns Radians
uw_init = uw_init1 /(2*pi*(echo[2]-echo[1]))

# B0 map
fmap = b0map(images,echo; smap = nothing, l2b, precon, finit = uw_init, mask = mask)[1]
pp = jim(fmap; title = "Field Map")
display(pp)

## Shim calculations

# Loss (objective) function for optimization.
# The field map f is modeled as f = H*A*s + f0, where
#   s = shim amplitudes (vector),
#   H = spherical harmonic basis functions
#   A = matrix containing shim coil expansion coefficients for basis in H
#   f0 = baseline field map at mask locations (vector)
loss = (s, HA, f0) -> norm(HA*s + f0, 2)^2 / length(f0)

ftol_rel = 0.5e-5


matf = matread("shimcal.mat");
F = matf["F"]   # [nx ny nz 8] for 2nd order shim systems
S = matf["S"]   # [8 8] matrix with shim amplitudes (typically diagonal)
mask_c = matf["mask_c"];
fov = matf["FOV_c"];

mask_c = BitArray(mask_c)

if l < 2
	F = F[:,:,:,1:3]
	ss = diag(S)
	S = Diagonal(ss[1:3])
end

inds = [3, 1, 2, 4, 6, 8, 7, 5]
Fr = copy(F)
for ii = 1:size(F,4)
	Fr[:,:,:,ii] = F[:,:,:,inds[ii]]
end
F = Fr

N = sum(vec(mask_c))

(nx,ny,nz,nShim) = size(F)

(x,y,z) = LinRange.(1, -1, [nx,ny,nz]) .* vec(fov)/2

# mask F and reshape to [N 8]
Fm = zeros(N, nShim)
for ii = 1:nShim
	f1 = F[:,:,:,ii]
	Fm[:,ii] = f1[mask_c]
end

# Get spherical harmonic basis of degree l
H = getSHbasis(x, y, z; L=l) # [nx ny nz numSH(l)]
H = reshape(H, :, size(H,4))
H = H[vec(mask_c), :] # size is [N sum(2*(0:l) .+ 1)]

# Get calibration matrix A
A = getcalmatrix(Fm, H, diag(S))

f0m = fmap[mask]

N = sum(vec(mask))

(x,y,z) = LinRange.(1, -1, [nx,ny,nz]) .* vec(fov)/2 ######

H = getSHbasis(x, y, z; L=l)
H = reshape(H, :, size(H,4))
H = H[vec(mask), :]
W = Diagonal(ones(N,))   # optional spatial weighting

s0 = -(W*H*A)\(W*f0m)    # Unconstrained least-squares solution (for comparison)

# This is where it all happens.
# @time shat = shimoptim(W*H*A, W*f0m, shimlims; loss=loss, ftol_rel=ftol_rel)
shat = s0

shat_ge = Int.(round.(shat))
shat_siemens = round.(shat; digits=1)

println("\nRecommended shim changes:")

println(string(
	"\tcf, x, y, z = ",
	shat_ge[1], ", ",
	shat_ge[3], ", ",
	shat_ge[4], ", ",
	shat_ge[2]))

if length(shat) > 4
	println(string("\t",
		"z2 ", shat_ge[5],
		" zx ", shat_ge[6],
		" zy ", shat_ge[7],
		" x2y2 ", shat_ge[8],
		" xy ", shat_ge[9]))

println(" ")
println(string(
	"GE: ",
	"\tset cf, x, y, z shims in Manual Prescan"))
	println(string(
		"\tsetNavShimCurrent",
		" z2 ", shat_ge[5],
		" zx ", shat_ge[6],
		" zy ", shat_ge[7],
		" x2y2 ", shat_ge[8],
		" xy ", shat_ge[9]))
	println(" ")
	println(string(
		"Siemens: adjvalidate -shim -set -mp -delta ",
		shat_siemens[3], " ",
		shat_siemens[4], " ",
		shat_siemens[2], " ",
		shat_siemens[5], " ",
		shat_siemens[6], " ",
		shat_siemens[7], " ",
		shat_siemens[8], " ",
		shat_siemens[9]))
else
	println(string(
		"\tSiemens: adjvalidate -shim -set -mp -delta ",
		shat_siemens[3], " ",
		shat_siemens[4], " ",
		shat_siemens[2]))
end

# predicted fieldmap after applying shims
fp = zeros(size(fmap))
fpm = H*A*shat + f0m
embed!(fp, fpm, mask)

# println()

# display predicted fieldmap
p = jim(log.(abs.(A[:,:]')); color=:jet)
p = jim(fp; clim=(-200,200), color=:jet)
p = jim(cat(fmap[:,:,:],fp[:,:,:];dims=1); ncol=6, clim=(-200,200), color=:jet)
display(p)

# predicted fieldmap after applying shims, no mask
fpnm = zeros(size(fmap));
H = getSHbasis(x, y, z; L=l);
H = reshape(H, :, size(H,4));
fpnmv = H*A*shat;
embed!(fpnm, fpnmv, BitArray(ones(nx,ny,nz)));
fpnm = fpnm + fmap;
println("Done")
#matwrite("nov8_f0z2.mat", Dict("f0" => fmap, "FOV" => fov))
#matwrite("nov8_mask1.mat", Dict("mask" => mask))