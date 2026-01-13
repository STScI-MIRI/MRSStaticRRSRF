from mrs_helpers import correct_miri_mrs_spectral_leak

if __name__ == "__main__":  # pragma: no cover

    # fmt: off
    #   info is ([short, medium, long], model file, type)
    sinfo = {"10Lac_center": (["jw03779002001_0310a", "jw03779004001_12101", "jw03779006001_0310a"], "10lac_mod_006_r10000.fits", "hot", "indigo"),
             }
    # fmt: on

    for m, cname in enumerate(sinfo.keys()):
        pname = cname
        cfiles, mfile, stype, scol = sinfo[cname]
        for k, cdith in enumerate(["1", "2", "3", "4"]):
            file_3a = f"{cname}/{cfiles[0]}_0000{cdith}_mirifulong_0_x1d.fits"
            file_1b = f"{cname}/{cfiles[1]}_0000{cdith}_mirifushort_0_x1d.fits"
            print(file_3a, file_1b)

            correct_miri_mrs_spectral_leak(
                file_3a, file_1b, "MRSStaticRRSRF/leak/MRS_spectral_leak_fractional.fits"
            )

