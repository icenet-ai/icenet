# Changelog

All notable changes to this project will be documented in this file.

## [v0.2.9] - 2025-02-19

### Added
- Plotting updates
    - Enable specifying CRS projection in forecast image/video plots.
    - Added functionality to specify latitude/longitude bounds for region selection instead of just pixel bounds.
    - Generate reference plot showing lon/lat region selected.
    - Added option to clip forecast plots to specified lat/lon subregions
    - Include lon/lat gridlines overlay.
- Documentation updates
    - Multiversion document generation (i.e., Can generate for historical releases, and latest dev branches).
    - Add icenet-notebook as submodule.
    - Integrates `icenet-notebooks` repo as tutorial guides within the Sphinx docs output.
- Auto-select if UNet model should use legacy rounding when prediction with a model trained using.
- Functions to use `pyproj` for reprojecting to various CRS.
- Use newer eccodes python package which now includes binary as well.
- Use `imageio-ffmpeg` for ffmpeg video output generation, can avoid system dependency.
- Add reference conda environment.yml file.

### Changed
- Modified reprojection logic to handle different CRS and leadtimes.
- Update Masking to handle lon/lat subregions.
- Documentation styling overhaul, and ready to be deployed on icenet website.
- Temporarily updated minimum eccodes version requirement to 2.37.0-2.38.3 (newer versions seeminly missing required binary).
- Enable forecast videos to be playable on stock Windows without added codecs.
- Get both the ensemble mean and member predictions (Useful for SIPN prediction).

### Fixed
- ORAS5 downloader not working: Migrate from MOTU to Copernicus Marine toolbox.
- Coastlines not working with forecast video generation.
- ERA5 download bug during first week of year.
- ERA5 downloader randomly not updating downloaded dataset with latest data.
- Forecast video output failing if ffmpeg missing and matplotlib fallsback to Pillow.

### Deprecated
- Removed deprecated CDS Download toolbox ERA5 downloader in favor of new CDSAPI.

### Removed
- Removed redundant code and functions during refactoring.
- Removed Python 3.8 support in favor of 3.9-3.12 Python versions.


### Other
- Various minor optimisations and bug fixes across plotting functions.
