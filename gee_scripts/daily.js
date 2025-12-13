// =======================================================
// INDIA DAILY FIRE DATASET (2018–2024)
// DAILY • CAUSAL • ML-READY • STATIC + HUMAN FACTORS
// =======================================================

// ---------------- REGION ----------------
var india = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "India"));

// ---------------- SETTINGS ----------------
var SCALE = 5000;     // 5 km
var START_YEAR = 2018;
var END_YEAR   = 2024;

// =======================================================
// STATIC / HUMAN LAYERS (REFERENCE YEAR ~2020)
// =======================================================

// Elevation (SRTM)
var elevation = ee.Image("USGS/SRTMGL1_003")
  .select("elevation");

// Population density (GPW v4)
var population = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
  .filterDate("2020-01-01", "2020-12-31")
  .first()
  .select("population_density");

// Forest fraction (MODIS IGBP)
var landcover = ee.ImageCollection("MODIS/061/MCD12Q1")
  .filterDate("2020-01-01", "2020-12-31")
  .first()
  .select("LC_Type1");

// Forest classes
var forestMask = landcover
  .eq(1).or(landcover.eq(2))
  .or(landcover.eq(3)).or(landcover.eq(4))
  .or(landcover.eq(5));

// ---- Aggregate static layers to 5 km ----
var elevationAgg = elevation
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject("EPSG:4326", null, SCALE)
  .rename("elevation");

var populationAgg = population
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject("EPSG:4326", null, SCALE)
  .rename("population_density");

var forestFraction = forestMask
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject("EPSG:4326", null, SCALE)
  .rename("forest_fraction");

// =======================================================
// DYNAMIC DATASETS
// =======================================================

// Burned Area (MODIS)
var fireIC = ee.ImageCollection("MODIS/061/MCD64A1");

// NDVI (MODIS)
var ndviIC = ee.ImageCollection("MODIS/061/MOD13Q1")
  .select("NDVI")
  .map(function(img) {
    return img
      .divide(10000)
      .rename("NDVI")
      .set("system:time_start", img.get("system:time_start"));
  });

// ERA5 Land Hourly
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY");

// =======================================================
// YEAR LOOP
// =======================================================
ee.List.sequence(START_YEAR, END_YEAR).evaluate(function(years) {

  years.forEach(function(YEAR) {

    print("Preparing year:", YEAR);

    var start = ee.Date.fromYMD(YEAR, 1, 1);
    var end   = start.advance(1, "year");
    var nDays = end.difference(start, "day");

    // ---------------- DAILY IMAGE COLLECTION ----------------
    var dailyIC = ee.ImageCollection(
      ee.List.sequence(0, nDays.subtract(1)).map(function(d) {

        var date = start.advance(d, "day");
        var next = date.advance(1, "day");

        // ===== FIRE (SAFE) =====
        var fireDayIC = fireIC
          .filterDate(date, next)
          .select("BurnDate");

        var fireCount = ee.Image(
          ee.Algorithms.If(
            fireDayIC.size().gt(0),
            fireDayIC.map(function(img) { return img.gt(0); }).sum(),
            ee.Image.constant(0)
          )
        ).rename("fire_count");

        var fireDetected = fireCount.gt(0).rename("fire_detected");

        // ===== WEATHER =====
        var eraDay = era5.filterDate(date, next);

        var tmean = eraDay.select("temperature_2m").mean()
          .rename("temperature_2m");

        var rain = eraDay.select("total_precipitation").sum()
          .rename("total_precipitation");

        var soil = eraDay.select("volumetric_soil_water_layer_1").mean()
          .rename("soil_moisture");

        var u10 = eraDay.select("u_component_of_wind_10m").mean();
        var v10 = eraDay.select("v_component_of_wind_10m").mean();
        var wind = u10.hypot(v10).rename("wind_speed");

        // ===== HUMIDITY =====
        var Td = eraDay.select("dewpoint_temperature_2m").mean();
        var T  = tmean;

        var Es = T.subtract(273.15)
          .multiply(17.625)
          .divide(T.subtract(273.15).add(243.04))
          .exp()
          .multiply(0.61094);

        var Ea = Td.subtract(273.15)
          .multiply(17.625)
          .divide(Td.subtract(273.15).add(243.04))
          .exp()
          .multiply(0.61094);

        var rh = Ea.divide(Es).multiply(100)
          .rename("relative_humidity");

        // ===== NDVI (PAST 16 DAYS ONLY — CAUSAL) =====
        var ndvi = ndviIC
          .filterDate(date.advance(-16, "day"), date)
          .mean()
          .rename("NDVI");

        // ===== FINAL DAILY IMAGE =====
        return ee.Image.cat([
          // Static (human + terrain)
          elevationAgg,
          populationAgg,
          forestFraction,

          // Dynamic
          tmean,
          rain,
          soil,
          wind,
          rh,
          ndvi,
          fireCount,
          fireDetected
        ])
        .set("date", date.format("YYYY-MM-dd"))
        .set("year", YEAR)
        .set("doy", date.getRelative("day", "year"));
      })
    );

    // ---------------- SAMPLE ----------------
    var sampled = dailyIC.map(function(img) {

      var fc = img.sample({
        region: india,
        scale: SCALE,
        geometries: true
      });

      return fc.map(function(f) {
        var geom = f.geometry();
        var coords = geom.coordinates();

        return f
          .set("latitude", coords.get(1))
          .set("longitude", coords.get(0))
          .set("date", img.get("date"))
          .set("year", img.get("year"))
          .set("doy", img.get("doy"));
      });
    }).flatten();

    // ---------------- EXPORT ----------------
    Export.table.toDrive({
      collection: sampled,
      description: "India_Daily_Fire_Static_" + YEAR,
      fileNamePrefix: "India_Daily_Fire_Static_" + YEAR,
      fileFormat: "CSV"
    });

  });
});
