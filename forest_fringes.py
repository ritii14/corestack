
import json
import ee

ee.Initialize(project='<project_id>')

# -------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------
TREE_CLASS = 6
FRINGE_WIDTH = 50
OUTER_BUFFER = 100
SCALE = 30
MAXPIX = 1e12

name = "asnawar"
input_file = f"{name}_mws_layers_features.json"


# -------------------------------------------------------
# LOAD MICROWATERSHEDS
# -------------------------------------------------------
with open(input_file) as f:
    geojson = json.load(f)

mws_fc = ee.FeatureCollection(geojson)

# -------------------------------------------------------
# TREE MODE (FOREST BASE)
# -------------------------------------------------------
lulc_imgs = ee.ImageCollection([
    ee.Image('projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2017_2018'),
    ee.Image('projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2018_2019'),
    ee.Image('projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2019_2020')
]).map(lambda img: img.select('predicted_label').eq(TREE_CLASS))

tree_mode = lulc_imgs.reduce(ee.Reducer.mode()).selfMask()
pixel_area = ee.Image.pixelArea()

# -------------------------------------------------------
# CHANGE PRODUCTS
# -------------------------------------------------------
ltp_change = ee.ImageCollection(
    'projects/corestack-datasets/assets/datasets/tree_health/final_ltp_stp_change_2017_2021'
).mean()

overall_change = ee.ImageCollection(
    'projects/corestack-trees/assets/tree_characteristics/overall_change_2017_2022'
).mean()

# -------------------------------------------------------
# PER-MWS METRICS
# -------------------------------------------------------
def compute_metrics_per_mws(f):
    mws_geom = f.geometry()
    mws_area = mws_geom.area(1)

    expanded_mws = mws_geom.buffer(OUTER_BUFFER, 1)

    forest = tree_mode.clip(expanded_mws)

    forest_patches = forest.reduceToVectors(
        geometry=expanded_mws,
        scale=SCALE,
        geometryType='polygon',
        eightConnected=True,
        maxPixels=MAXPIX
    )

    ltps = forest_patches.filter(ee.Filter.area(10000, 1e13))

    def make_fringe(p):
        outer = p.geometry()
        inner = outer.buffer(-FRINGE_WIDTH, 1)
        return ee.Feature(outer.difference(inner, 1))

    fringes = ltps.map(make_fringe)

    fringes_clipped = fringes.map(
        lambda fr: ee.Feature(
            fr.geometry().intersection(mws_geom, 1)
        )
    )

    fringe_img = ee.Image.constant(1) \
        .paint(fringes_clipped, 1) \
        .selfMask() \
        .clip(mws_geom)

    fringe_area = fringes_clipped.geometry().area(1)


    # ---- Deforestation & degradation masks
    deforestation = ltp_change.eq(6).Or(ltp_change.eq(7)) \
        .updateMask(tree_mode) \
        .clip(mws_geom)

    degradation = overall_change.eq(-1) \
        .updateMask(tree_mode) \
        .clip(mws_geom)

    # ---- Areas in MWS
    defo_mws_area = pixel_area.updateMask(deforestation).reduceRegion(
        ee.Reducer.sum(), mws_geom, SCALE, maxPixels=MAXPIX
    ).get('area')

    degr_mws_area = pixel_area.updateMask(degradation).reduceRegion(
        ee.Reducer.sum(), mws_geom, SCALE, maxPixels=MAXPIX
    ).get('area')

    fringe_geom = fringes_clipped.geometry()

    fringe_area = ee.Number(
        ee.Algorithms.If(
            fringe_geom.area(1).eq(0),
            0,
            fringe_geom.area(1)
        )
    )

    fringe_is_empty = fringe_area.eq(0)

    deforestation_fringe = ee.Image(
        ee.Algorithms.If(
            fringe_is_empty,
            ee.Image(0).updateMask(ee.Image(0)),
            deforestation.clip(fringe_geom)
        )
    )

    degradation_fringe = ee.Image(
        ee.Algorithms.If(
            fringe_is_empty,
            ee.Image(0).updateMask(ee.Image(0)),
            degradation.clip(fringe_geom)
        )
    )


    defo_fringe_area = pixel_area.updateMask(deforestation_fringe).reduceRegion(
        ee.Reducer.sum(), fringe_geom, SCALE, maxPixels=MAXPIX
    ).get('area')

    degr_fringe_area = pixel_area.updateMask(degradation_fringe).reduceRegion(
        ee.Reducer.sum(), fringe_geom, SCALE, maxPixels=MAXPIX
    ).get('area')


    # ---- Ratios
    fringe_to_mws_ratio = ee.Number(fringe_area).divide(mws_area)
    degr_fringe_ratio = ee.Number(degr_fringe_area).divide(fringe_area)
    defo_fringe_ratio = ee.Number(defo_fringe_area).divide(fringe_area)

    return ee.Feature(f.geometry()).set({
    'uid': f.get('uid'),

    'mws_area_m2': mws_area,
    'forest_fringe_area_m2': fringe_area,
    'forest_fringe_ratio': fringe_to_mws_ratio,

    'tree_degradation_mws_area_m2': degr_mws_area,
    'tree_degradation_fringe_area_m2': degr_fringe_area,
    'tree_degradation_fringe_ratio': degr_fringe_ratio,

    'tree_deforestation_mws_area_m2': defo_mws_area,
    'tree_deforestation_fringe_area_m2': defo_fringe_area,
    'tree_deforestation_fringe_ratio': defo_fringe_ratio
})


# -------------------------------------------------------
# RUN FOR ALL MWS
# -------------------------------------------------------
results_fc = mws_fc.map(compute_metrics_per_mws)

final_fc = results_fc.select([
    'uid',
    'mws_area_m2',
    'forest_fringe_area_m2',
    'forest_fringe_ratio',
    'tree_degradation_mws_area_m2',
    'tree_degradation_fringe_area_m2',
    'tree_degradation_fringe_ratio',
    'tree_deforestation_mws_area_m2',
    'tree_deforestation_fringe_area_m2',
    'tree_deforestation_fringe_ratio'
])


def extract_fringes(f):
    mws_geom = f.geometry()
    expanded_mws = mws_geom.buffer(OUTER_BUFFER, 1)
    forest = tree_mode.clip(expanded_mws)

    forest_patches = forest.reduceToVectors(
        geometry=expanded_mws,
        scale=SCALE,
        geometryType='polygon',
        eightConnected=True,
        maxPixels=MAXPIX
    )

    ltps = forest_patches.filter(ee.Filter.area(10000, 1e13))

    def make_fringe(p):
        outer = p.geometry()
        inner = outer.buffer(-FRINGE_WIDTH, 1)
        return ee.Feature(
            outer.difference(inner, 1)
                 .intersection(mws_geom, 1),
            {'mws_id': f.get('id')}
        )

    return ltps.map(make_fringe)

# -------------------------------------------------------
# EXPORT
# -------------------------------------------------------
task = ee.batch.Export.table.toDrive(
    collection=final_fc,
    description='Bichhiya_MWS_ForestFringe_Corrected',
    folder='<folder_name>',
    fileNamePrefix=f'{name}_forest_fringe_metrics_corrected',
    fileFormat='CSV'
)

task.start()
print("Export started: mws_forest_fringe_metrics_corrected.csv")
