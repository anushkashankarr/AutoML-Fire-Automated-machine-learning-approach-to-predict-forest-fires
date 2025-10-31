// === ERA5-Land Weather Variables (2018–2024) ===

// Load WWF ecozones (simplified for performance)
var teow_in_india = ee.FeatureCollection('projects/driven-seer-401816/assets/wwf')
  .map(function(f) {
    return f.simplify(2000); // simplify to ~2 km
  });

// Load ERA5-Land Daily Aggregates
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .select([
    'temperature_2m',
    'total_precipitation_sum',
    'dewpoint_temperature_2m',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m'
  ]);

// === Derived Variables ===
function processERA5(img) {
  var temp = img.select('temperature_2m').subtract(273.15).rename('temp_mean');
  var dew = img.select('dewpoint_temperature_2m').subtract(273.15);

  // Relative Humidity (using Magnus formula)
  var rh = dew.expression(
    '100 * (exp((17.625 * dew)/(243.04 + dew)) / exp((17.625 * temp)/(243.04 + temp)))',
    { dew: dew, temp: temp }
  ).rename('rel_humidity');

  // Wind speed magnitude
  var wind = img.expression('sqrt(u*u + v*v)', {
    u: img.select('u_component_of_wind_10m'),
    v: img.select('v_component_of_wind_10m')
  }).rename('wind_speed');

  // Rainfall (mm)
  var rain = img.select('total_precipitation_sum').multiply(1000).rename('rainfall_mm');

  return temp.addBands([rain, rh, wind]).set('system:time_start', img.get('system:time_start'));
}

// Apply processing function to all ERA5 images
era5 = era5.map(processERA5);

// === Define Year and Month Ranges ===
var years = ee.List.sequence(2018, 2024);
var months = ee.List.sequence(1, 12);

// === Loop over each year and export monthly means per ecozone ===
years.getInfo().forEach(function(y) {
  var year = ee.Number(y);
  print('🚀 Starting export for year:', year);

  var monthly = months.map(function(m) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var monthlyImg = era5.filterDate(start, end).mean();

    // Reduce per ecozone
    var stats = monthlyImg.reduceRegions({
      collection: teow_in_india,
      reducer: ee.Reducer.mean(),
      scale: 9000
    }).map(function(f) {
      return f.set({
        'year': year,
        'month': m,
        'start': start.format('YYYY-MM-dd'),
        'end': end.format('YYYY-MM-dd')
      });
    });

    return stats;
  }).flatten();

  // Export as yearly CSV
  Export.table.toDrive({
    collection: ee.FeatureCollection(monthly).flatten(),
    description: 'ERA5_Weather_by_Ecozone_' + year.format(),
    folder: 'AutoML-Fire-Exports',
    fileNamePrefix: 'ERA5_Weather_by_Ecozone_' + year.format(),
    fileFormat: 'CSV'
  });
});
