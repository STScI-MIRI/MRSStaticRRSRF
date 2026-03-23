import glob
import argparse
from MRS_PFPC.utils.mrs_helpers import subdithers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star")
    args = parser.parse_args()

    # name of star
    starname = args.starname

    main_path = f"{starname}/"

    # all dither 1 files
    files = glob.glob(f"{main_path}/jw*_00001_*mirifushort_rate.fits") + glob.glob(
        f"{main_path}/jw*_00001_*mirifulong_rate.fits"
    )
    ratefiles = sorted(files)
    print(ratefiles)
    exit()

    subdithers(ratefiles)


if __name__ == "__main__":
    main()
