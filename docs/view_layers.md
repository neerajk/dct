# View-only layers (QGIS / browser)

**Rule:** these are for looking at and showing sites. They are never downloaded, cached
or used as training input. DCT code contains no Esri fetcher on purpose.

## Esri World Imagery (zoom 17-18)
- QGIS: Layer > Add Layer > XYZ Tiles, URL
  `https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`
  Max zoom 18 (z18 ~0.57 m/px at lat 18 N, ~0.37 m/px at lat 52 N; z17 is double).
- Capture dates and real source resolution vary by location (India is often older/coarser than the
  zoom suggests). Check the "World Imagery (Clarity) / Metadata" layer or the QGIS "Esri Wayback"
  plugin for dated releases.
- Esri item terms: not intended for exporting tiles for offline use; Esri Master License Agreement applies.
  For client reports get written confirmation from Esri or use an ArcGIS Location Platform account.
- Always show: "Powered by Esri - Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community".

## PDOK aerial photos (open, CC BY 4.0, credit Kadaster / Beeldmateriaal Nederland)
- WMTS: `https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts`
- Layers: `2016_ortho25` ... `2020_ortho25`, `2021_orthoHR` ... `2026_orthoHR`, `Actueel_orthoHR`.

## Hessen DOP20 (dl-zero-de/2.0)
- WMS: `https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows?SERVICE=WMS&REQUEST=GetCapabilities`
- Layer `he_dop20_rgb` (RGB), `he_dop20_cir` (colour infrared). Current edition only.

## Google Earth
Open `outputs/dct_sites_verify.kml` (Google Earth Pro: File > Open; Earth Web: Projects > Import KML).
Use the History slider (clock icon) to see older imagery for each site.
