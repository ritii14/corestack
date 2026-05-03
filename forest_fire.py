
import ee
import json
import geemap

ee.Initialize(project='<project_id>')

SCALE = 1000
MAXPIX = 1e13

name = "keonjhar"  #add name of the block here
input_file = f"{name}_mws_layers_features.json"  #administrative boundaries file downloaded from landscape explorer
TARGET_UID = "7_25388"  #targeted microwatershed

# ---------------------------------------
# TIME PARAMETERS
# ---------------------------------------
START_YEAR = 2001
END_YEAR   = 2022

START_DATE = f"{START_YEAR}-01-01"
END_DATE   = f"{END_YEAR}-12-31"

N_YEARS = END_YEAR - START_YEAR + 1

print(f"\nAnalysis Period: {START_YEAR}–{END_YEAR} ({N_YEARS} years)")

# ---------------------------------------
# LOAD MWS
# ---------------------------------------
with open(input_file) as f:
    geojson = json.load(f)

mws_fc = ee.FeatureCollection(geojson)

# ---------------------------------------
# LOAD FIRE DATA
# ---------------------------------------
terra = ee.ImageCollection("MODIS/061/MOD14A1")
aqua  = ee.ImageCollection("MODIS/061/MYD14A1")

fires = terra.merge(aqua).filterDate(START_DATE, END_DATE)
frp_collection = fires.select('MaxFRP')

# ---------------------------------------
# PREPROCESSING
# ---------------------------------------
def mask_fire(img):
    return img.updateMask(img.gt(0))

frp_masked = frp_collection.map(mask_fire)

def fire_binary(img):
    return img.gt(0).unmask(0).rename('fire')

fire_binary_collection = frp_collection.map(fire_binary)

# ---------------------------------------
# GENERIC PIPELINE FUNCTION
# ---------------------------------------
def run_pipeline(mode, output_name):

    if mode == "sum":
        img = frp_masked.sum().divide(N_YEARS)   
        reducer = ee.Reducer.sum()
        band = 'MaxFRP'

    elif mode == "mean":
        img = frp_masked.mean() 
        reducer = ee.Reducer.mean()
        band = 'MaxFRP'

    elif mode == "max":
        img = frp_masked.max() 
        reducer = ee.Reducer.max()
        band = 'MaxFRP'

    elif mode == "count":
        img = fire_binary_collection.sum().divide(N_YEARS) 
        reducer = ee.Reducer.sum()
        band = 'fire'

    else:
        raise ValueError("Invalid mode")

    def compute(feature):
        value = img.reduceRegion(
            reducer=reducer,
            geometry=feature.geometry(),
            scale=SCALE,
            maxPixels=MAXPIX
        ).get(band)

        value = ee.Number(ee.Algorithms.If(value, value, 0))

        return feature.set({mode: value})

    fc = mws_fc.map(compute)

    percentiles = fc.reduceColumns(
        reducer=ee.Reducer.percentile([10, 25, 40, 55, 70, 85, 95]),
        selectors=[mode]
    )

    p = percentiles.getInfo()

    p10 = p['p10']
    p25 = p['p25']
    p40 = p['p40']
    p55 = p['p55']
    p70 = p['p70']
    p85 = p['p85']

    def classify(feature):
        value = ee.Number(feature.get(mode))

        color = ee.Algorithms.If(value.eq(0), '#ffffff',
                ee.Algorithms.If(value.lt(p10), '#ffffcc',
                ee.Algorithms.If(value.lt(p25), '#ffeda0',
                ee.Algorithms.If(value.lt(p40), '#fed976',
                ee.Algorithms.If(value.lt(p55), '#feb24c',
                ee.Algorithms.If(value.lt(p70), '#fd8d3c',
                ee.Algorithms.If(value.lt(p85), '#f03b20',
                                 '#bd0026')))))))

        return feature.set({
            'style': {
                'color': 'black',
                'width': 1,
                'fillColor': color
            }
        })

    styled_fc = fc.map(classify)

    target_mws = mws_fc.filter(ee.Filter.eq('uid', TARGET_UID))
    target_feature = fc.filter(ee.Filter.eq('uid', TARGET_UID)).first()
    target_value = target_feature.get(mode).getInfo()

    print("\n==============================")
    print(f"MODE: {mode.upper()} (per year)")
    print("==============================")

    print(f"Target UID ({TARGET_UID}) value: {round(target_value, 2)}")

    print("\nClass thresholds:")
    print(f"0              → No Fire")
    print(f"0  - {round(p10,2)}  → Very Low")
    print(f"{round(p10,2)} - {round(p25,2)} → Low")
    print(f"{round(p25,2)} - {round(p40,2)} → Moderate")
    print(f"{round(p40,2)} - {round(p55,2)} → Moderately High")
    print(f"{round(p55,2)} - {round(p70,2)} → High")
    print(f"{round(p70,2)} - {round(p85,2)} → Very High")
    print(f"> {round(p85,2)} → Extreme")

    # ---- Map ----
    Map = geemap.Map(center=[23.5, 80], zoom=7)

    Map.addLayer(
        styled_fc.style(**{'styleProperty': 'style'}),
        {},
        f"{mode.upper()} (per year)"
    )

    Map.addLayer(
        target_mws.style(color='blue', width=3, fillColor='00000000'),
        {},
        'Selected UID'
    )

    Map.to_html(output_name)
    print(f"{mode} map saved → {output_name}")


# ---------------------------------------
# RUN ALL
# ---------------------------------------
run_pipeline("sum",   "sum_frp_per_year.html")
run_pipeline("mean",  "mean_frp.html")
run_pipeline("max",   "max_frp.html")
run_pipeline("count", "fire_count_per_year.html")
