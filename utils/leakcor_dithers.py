from mrs_helpers import correct_miri_mrs_spectral_leak

if __name__ == "__main__":  # pragma: no cover

    # fmt: off
    sinfo = {"muCol": (["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"], "mucol_mod_006_r10000.fits"),
             "delUMi": (["jw01536024001_04102", "jw01536024001_04104", "jw01536024001_04106"], "delumi_mod_005_r10000.fits"),
             "HR5467": (["jw04496009001_03102", "jw04496009001_03104", "jw04496009001_03106"], "hd128998_mod_004_r10000.fits"),
             "HD2811_c1": (["jw01536022001_08101", "jw01536022001_06101", "jw01536022001_06101"], "hd2811_mod_006_r10000.fits"),
             "HD2811_c2": (["jw04496002001_04106", "jw04496002001_04104", "jw04496002001_04102"], "hd2811_mod_006_r10000.fits"),
             "HD2811_c3": (["jw06604006001_08101", "jw06604006001_06101", "jw06604006001_04101"], "hd2811_mod_006_r10000.fits"),
             "Athalia": (["jw01549006001_04106", "jw01549006001_04104", "jw01549006001_04102"], None)
             }
    # fmt: on

    for m, cname in enumerate(sinfo.keys()):
        pname = cname
        cfiles, mfile = sinfo[cname]
        for k, cdith in enumerate(["1", "2", "3", "4"]):
            file_3a = f"{cname}/{cfiles[0]}_0000{cdith}_mirifulong_0_x1d.fits"
            file_1b = f"{cname}/{cfiles[1]}_0000{cdith}_mirifushort_0_x1d.fits"
            print(file_3a, file_1b)

            correct_miri_mrs_spectral_leak(
                file_3a, file_1b, "leak/MRS_spectral_leak_fractional.fits"
            )

