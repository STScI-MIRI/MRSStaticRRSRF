from mrs_helpers import correct_miri_mrs_spectral_leak

if __name__ == "__main__":  # pragma: no cover

    # fmt: off
    #   info is ([short, medium, long], model file, type)
    sinfo = {"muCol": (["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"], "mucol_mod_006_r10000.fits", "hot"),
             "delUMi": (["jw01536024001_04102", "jw01536024001_04104", "jw01536024001_04106"], "delumi_mod_005_r10000.fits", "A"),
             "HR5467": (["jw04496009001_03102", "jw04496009001_03104", "jw04496009001_03106"], "hd128998_mod_004_r10000.fits", "A"),
             "HD2811_c1": (["jw01536022001_08101", "jw01536022001_06101", "jw01536022001_06101"], "hd2811_mod_006_r10000.fits", "A"),
             "HD2811_c2": (["jw04496002001_04106", "jw04496002001_04104", "jw04496002001_04102"], "hd2811_mod_006_r10000.fits", "A"),
             "HD2811_c3": (["jw06604006001_08101", "jw06604006001_06101", "jw06604006001_04101"], "hd2811_mod_006_r10000.fits", "A"),
             "HD2811_c4": (["jw07487105001_03106", "jw07487105001_03104", "jw07487105001_03102"], "hd2811_mod_006_r10000.fits", "A"),
             "16CygB": (["jw01538001001_03102", "jw01538001001_03104", "jw01538001001_03106"], "16cygb_mod_005_r10000.fits", "G"),
             "Athalia": (["jw01549006001_04106", "jw01549006001_04104", "jw01549006001_04102"], None, "asteroid"),
             "Jena": (["jw01549055001_04106", "jw01549055001_04104", "jw01549055001_04102"], None, "asteroid"),
             "HD283809": (["jw04551001001_03102", "jw04551001001_03104", "jw04551001001_03106"], None, None),
             "2MASSJ150958": [["jw02183038001_09101", "jw02183038001_07101", "jw02183038001_05101"], None, None],
             }
    # fmt: on

    for m, cname in enumerate(sinfo.keys()):
        pname = cname
        cfiles, mfile, stype = sinfo[cname]
        for k, cdith in enumerate(["1", "2", "3", "4"]):
            file_3a = f"{cname}/{cfiles[0]}_0000{cdith}_mirifulong_0_x1d.fits"
            file_1b = f"{cname}/{cfiles[1]}_0000{cdith}_mirifushort_0_x1d.fits"
            print(file_3a, file_1b)

            correct_miri_mrs_spectral_leak(
                file_3a, file_1b, "leak/MRS_spectral_leak_fractional.fits"
            )

