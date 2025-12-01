// =======================================================
// INDIA FIRE INTENSITY DATASET (2018–2024)
// Continuous fire intensity feature
// =======================================================

// 1️⃣ REGION
var india = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
  .filter(ee.Filter.eq('ADM0_NAME', 'India'));
var SCALE = 10000;
var TARGET_PER_CLASS = 2500;

// 2️⃣ LAND MASK
var lc = ee.ImageCollection("MODIS/061/MCD12Q1")
  .filterDate('2018-01-01','2024-12-31')
  .first()
  .select('LC_Type1');
var landMask = lc.neq(0).unmask(0);

// 3️⃣ FIRE INTENSITY (continuous)
var fireCount = ee.ImageCollection("MODIS/061/MCD64A1")
  .filterDate('2018-01-01','2024-12-31')
  .select('BurnDate')
  .map(function(img){ return img.gt(0).unmask(0); })
  .sum()
  .rename('fire_intensity');

// fire ever mask
var fireEver = fireCount.gt(0).rename('fireEver');

// Define masks
var fireMask = fireEver.eq(1);
var nonFireMask = fireEver.eq(0).and(landMask);

// 4️⃣ WEATHER VARIABLES (mean 7 years)
var eraLand = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .filterDate('2018-01-01','2024-12-31')
  .select([
    'temperature_2m_max','temperature_2m_min','total_precipitation_sum',
    'u_component_of_wind_10m','v_component_of_wind_10m',
    'surface_solar_radiation_downwards_sum','volumetric_soil_water_layer_1'
  ])
  .mean().unmask(0);

var windspeed = eraLand.select('u_component_of_wind_10m')
  .hypot(eraLand.select('v_component_of_wind_10m'))
  .rename('windspeed');

var weather = eraLand.addBands(windspeed).rename([
  'tmax','tmin','rain','u_wind','v_wind',
  'solar_radiation','soil_moisture','windspeed'
]);

// HUMIDITY
var era = ee.ImageCollection("ECMWF/ERA5/DAILY")
  .filterDate('2018-01-01','2024-12-31')
  .select(['mean_2m_air_temperature','dewpoint_2m_temperature'])
  .mean().unmask(0);
var T  = era.select('mean_2m_air_temperature').subtract(273.15);
var Td = era.select('dewpoint_2m_temperature').subtract(273.15);
var es = T.multiply(17.67).divide(T.add(243.5)).exp().multiply(6.112);
var e  = Td.multiply(17.67).divide(Td.add(243.5)).exp().multiply(6.112);
var rh = e.divide(es).multiply(100).rename('humidity');

// NDVI
var ndvi = ee.ImageCollection("MODIS/061/MOD13Q1")
  .filterDate('2018-01-01','2024-12-31')
  .select('NDVI')
  .map(function(img){ return img.unmask(0); })
  .mean().rename('ndvi');

// CLOUD COVER
var cloud = ee.ImageCollection('MODIS/061/MOD09GA')
  .filterDate('2018-01-01','2024-12-31')
  .select('state_1km')
  .map(function(img){
    return img.bitwiseAnd(3).eq(2).rename('cloudcover');
  })
  .mean().multiply(100).rename('cloudcover');

// TERRAIN
var terrain = ee.Terrain.products(ee.Image("USGS/SRTMGL1_003"))
  .select(['elevation','slope','aspect']).unmask(0);

// FINAL VARIABLE STACK
var vars = weather.addBands([rh, cloud, ndvi, terrain, lc.rename('landcover'), fireCount]);

// ---------- STRATIFIED SAMPLING ----------
var varsWithClass = vars.addBands(fireEver.rename('fireClass'));

var sampled = varsWithClass.stratifiedSample({
  numPoints: 0,
  classBand: 'fireClass',
  classValues: [1, 0],
  classPoints: [TARGET_PER_CLASS, TARGET_PER_CLASS],
  region: india,
  scale: SCALE,
  seed: 42,
  geometries: true
});

// ---------- POSTPROCESS ----------
sampled = sampled.map(function(f){
  var cls = ee.Number(f.get('fireClass'));
  var intensity = ee.Number(f.get('fire_intensity'));
  // Fire intensity = 0 → no fire, >0 = number of burns
  return f.set('fire', cls).set('fire_intensity', intensity);
});

print('Fire points:', sampled.filter('fire == 1').size());
print('Non-fire points:', sampled.filter('fire == 0').size());
print('Total:', sampled.size());

// ---------- VISUALIZE ----------
Map.centerObject(india, 4);
Map.addLayer(fireCount, {min:0, max:10, palette:['white','orange','red']}, 'Fire Intensity');
Map.addLayer(sampled.style({color:'yellow', pointSize: 2}), {}, 'Sampled Points');

// ---------- EXPORT ----------
Export.table.toDrive({
  collection: sampled,
  description: 'India_FireIntensity_2018_2024',
  fileNamePrefix: 'India_FireIntensity_2018_2024',
  fileFormat: 'CSV'
});
