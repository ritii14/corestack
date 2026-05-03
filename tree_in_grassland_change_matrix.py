
import ee
import json
ee.Initialize(project='<project_id>')

# ----------------------------------------
# PARAMETERS
# ----------------------------------------

TREE_CLASS = 6
SHRUB_CLASS = 12
SHRUB_THRESHOLD = 0.5
RADIUS_M = 100
SCALE = 30
MAXPIX = 1e12

LULC_CLASSES = {
    0: 'Background',
    1: 'Built_up',
    2: 'Kharif_water',
    3: 'Kharif_Rabi_water',
    4: 'Kharif_Rabi_Zaid_water',
    5: 'Crops',
    6: 'Trees',
    7: 'Barren_land',
    8: 'Single_Kharif',
    9: 'Single_Non_Kharif',
    10: 'Double_Cropping',
    11: 'Triple_Annual_Perennial',
    12: 'Shrubs_Scrubs'
}

TREE_CLASS = 6
NEIGHBOR_CLASSES = [k for k in LULC_CLASSES.keys() if k != TREE_CLASS]
THRESHOLD = 0.5   # strictly > 50%
pixel_area = ee.Image.pixelArea()
name = "bichhiya"

# ----------------------------------------
# LOAD MICROWATERSHED
# ----------------------------------------
with open(f"{name}_mws_layers_features.json") as f:
    geojson = json.load(f)

mws_fc = ee.FeatureCollection(geojson)

# ----------------------------------------
# LOAD LULC IMAGES BY YEAR
# ----------------------------------------
lulc_by_year = {
    2018: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2017_2018'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2019: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2018_2019'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2020: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2019_2020'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2021: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2020_2021'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2022: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2021_2022'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2023: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2022_2023'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2024: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2023_2024'
    ).select('predicted_label')
     .unmask(0)
     .toInt(),

    2025: ee.Image(
        'projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/'
        'pan_india_lulc_v3_2024_2025'
    ).select('predicted_label')
     .unmask(0)
     .toInt()
}



def tree_context_all(lulc, aoi):
    kernel = ee.Kernel.circle(RADIUS_M, 'meters')
    lulc_img = lulc.clip(aoi.buffer(110))

    tree_mask  = lulc_img.eq(TREE_CLASS)
    shrub_mask = lulc_img.eq(12)

    total_px = ee.Image.constant(1) \
        .clip(aoi.buffer(110)) \
        .reduceNeighborhood(ee.Reducer.sum(), kernel)

    shrub_frac = shrub_mask.toInt() \
        .reduceNeighborhood(ee.Reducer.sum(), kernel) \
        .divide(total_px)

    # Tree embedded in shrubland
    tree_in_shrub = tree_mask.And(shrub_frac.gt(THRESHOLD))

    # Shrubs associated with those trees
    shrub_around_tree = shrub_mask.And(
        tree_in_shrub.focal_max(radius=RADIUS_M, units='meters')
    )

    return (
        ee.Image(0)
        .where(tree_in_shrub, 1)
        .where(shrub_around_tree, 2)
        .toInt()
        .clip(aoi)
    )

#range = 2018 to 2025
start_year = 2019
end_year = 2020

def temporal_context(aoi):
    ctx_start1 = tree_context_all(lulc_by_year[start_year-1], aoi)
    ctx_start2 = tree_context_all(lulc_by_year[start_year], aoi)
    ctx_start3 = tree_context_all(lulc_by_year[start_year+1], aoi)
    ctx_end1 = tree_context_all(lulc_by_year[end_year-1], aoi)
    ctx_end2 = tree_context_all(lulc_by_year[end_year], aoi)
    ctx_end3 = tree_context_all(lulc_by_year[end_year+1], aoi)    

    context_start = ee.ImageCollection([ctx_start1, ctx_start2, ctx_start3]) \
        .reduce(ee.Reducer.mode()).toInt()

    context_end = ee.ImageCollection([ctx_end1, ctx_end2, ctx_end3]) \
        .reduce(ee.Reducer.mode()).toInt()

    return context_start, context_end

def compute_A(f):
    aoi = f.geometry()
    uid = f.get('uid')

    lulc_start = ee.ImageCollection([
        lulc_by_year[start_year-1],
        lulc_by_year[start_year],
        lulc_by_year[start_year+1]
    ]).reduce(ee.Reducer.mode()).clip(aoi)

    lulc_end = ee.ImageCollection([
        lulc_by_year[end_year-1],
        lulc_by_year[end_year],
        lulc_by_year[end_year+1]
    ]).reduce(ee.Reducer.mode()).clip(aoi)


    context_start, context_end = temporal_context(aoi)


    tree_loss = ((context_start.eq(1)).Or(context_start.eq(2))).And(context_end.eq(0))
    tree_to_barren = ((context_start.eq(1)).Or(context_start.eq(2))).And(lulc_end.eq(7))

    def area(mask):
        return pixel_area.updateMask(mask).reduceRegion(
            ee.Reducer.sum(), aoi, SCALE, maxPixels=MAXPIX
        ).get('area')

    grassland_area = area((context_start.eq(1)) \
             .Or(context_start.eq(2)))
    tree_in_shrub_area = area(context_start.eq(1))
    isolated_shrub_area = area(lulc_start.eq(12).And(context_start.eq(0)))
    shrubland_area = area(lulc_start.eq(12))

    tree_loss_area = area(tree_loss)
    barren_area = area(tree_to_barren)

    return ee.Feature(None).set({
        'uid': uid,
        'grassland_area_m2': grassland_area,
        'tree_in_shrub_area_m2': tree_in_shrub_area,
        'isolated_shrub_area_m2': isolated_shrub_area,
        'shrubland_area_m2': shrubland_area,
        'tree_loss_area_m2': tree_loss_area,
        'tree_loss_to_grassland_ratio':
            ee.Number(tree_loss_area).divide(grassland_area),
        'tree_loss_to_tree_in_shrub_ratio':
            ee.Number(tree_loss_area).divide(tree_in_shrub_area),
        'tree_shrub_to_barren_area_m2': barren_area
    })

def compute_B(f):
    aoi = f.geometry()
    uid = f.get('uid')

    lulc_end = ee.ImageCollection([
        lulc_by_year[end_year-1],
        lulc_by_year[end_year],
        lulc_by_year[end_year+1]
    ]).reduce(ee.Reducer.mode()).clip(aoi)

    context_start, context_end = temporal_context(aoi)

    to_built = ((context_start.eq(1)).Or(context_start.eq(2))).And(lulc_end.eq(1))
    to_kharif = ((context_start.eq(1)).Or(context_start.eq(2))).And(lulc_end.eq(2))
    to_kharif_rabi = ((context_start.eq(1)).Or(context_start.eq(2))).And(lulc_end.eq(3))

    def area(mask):
        return pixel_area.updateMask(mask).reduceRegion(
            ee.Reducer.sum(), aoi, SCALE, maxPixels=MAXPIX
        ).get('area')

    return ee.Feature(None).set({
        'uid': uid,
        'tree_shrub_to_built_area_m2': area(to_built),
        'tree_shrub_to_kharif_water_area_m2': area(to_kharif),
        'tree_shrub_to_kharif_rabi_water_area_m2': area(to_kharif_rabi)
    })

def compute_C(f):
    aoi = f.geometry()
    uid = f.get('uid')
  
    lulc_end = ee.ImageCollection([
        lulc_by_year[end_year-1],
        lulc_by_year[end_year],
        lulc_by_year[end_year+1]
    ]).reduce(ee.Reducer.mode()).clip(aoi)

    context_start, context_end = temporal_context(aoi)

    to_zaid = ((context_start.eq(1)).Or(context_start.eq(2))).And(lulc_end.eq(4))
    to_crops = ((context_start.eq(1)).Or(context_start.eq(2))).And(
        lulc_end.eq(5)
        .Or(lulc_end.eq(8))
        .Or(lulc_end.eq(9))
        .Or(lulc_end.eq(10))
        .Or(lulc_end.eq(11))
    )

    def area(mask):
        return pixel_area.updateMask(mask).reduceRegion(
            ee.Reducer.sum(), aoi, SCALE, maxPixels=MAXPIX
        ).get('area')

    return ee.Feature(None).set({
        'uid': uid,
        'tree_shrub_to_kharif_rabi_zaid_water_area_m2': area(to_zaid),
        'tree_shrub_to_crops_area_m2': area(to_crops)
    })

results_C = mws_fc.map(compute_C)


results_B = mws_fc.map(compute_B)


results_A = mws_fc.map(compute_A)

task = ee.batch.Export.table.toDrive(
    collection=results_A,
    description='MWS_Tree_Shrub_Context_Metrics_A',
    folder='<folder_name>',
    fileNamePrefix=f'{name}_mws_tree_shrub_context_metrics_A',
    fileFormat='CSV'
)
task.start()
task = ee.batch.Export.table.toDrive(
    collection=results_B,
    description='MWS_Tree_Shrub_Context_Metrics_B',
    folder='<folder_name>',
    fileNamePrefix=f'{name}_mws_tree_shrub_context_metrics_B',
    fileFormat='CSV'
)
task.start()
task = ee.batch.Export.table.toDrive(
    collection=results_C,
    description='MWS_Tree_Shrub_Context_Metrics_C',
    folder='<folder_name>',
    fileNamePrefix=f'{name}_mws_tree_shrub_context_metrics_C',
    fileFormat='CSV'
)
task.start()


print("Export started: mws_tree_shrub_context_metrics.csv")
