
COLUMN_ORDER = [
    'id', 'listed_to_sold_days', 'first_desc', 'first_sold_date', 'first_sold_price', 
    'first_listed_date', 'first_listed_price', 'first_img_link', 'first_img_arr_list',
    'first_desc_date', 'latitude', 'longitude', 'floor_size', 'year_built', 'exterior',
    'num_rooms', 'land_value', 'roof_material', 'style', 'lot_size', 'additions_value', 
    'heating_fuel', 'heating_type', 'parking_spaces', 'cooling_system', 'foundation_details'
]
ID_COLS = ['id']
OUTCOME_COLS = ['listed_to_sold_days', 'first_sold_price']
TEXT_FEATURE = 'first_desc'
IMG_FEATURE = 'first_img_arr_list'
EXTRA_COLS = ['first_img_link', 'first_desc_date', 'first_sold_date']
TAB_FEATURES = [
    'latitude', 'longitude', 'floor_size', 'year_built', 'exterior', 'num_rooms',
    'land_value', 'roof_material', 'style', 'lot_size', 'additions_value',
    'heating_fuel', 'heating_type', 'parking_spaces', 'cooling_system', 
    'foundation_details', 'first_listed_date', 'first_listed_price'
]
RECONSTRUCT_COLS = [
    *ID_COLS, *OUTCOME_COLS, TEXT_FEATURE, IMG_FEATURE, *EXTRA_COLS, *TAB_FEATURES
] 

TAB_CAT_FEATURES = [
    'exterior', 'roof_material', 'style', 'heating_fuel', 'heating_type',
    'cooling_system', 'foundation_details'
]
TAB_DT_FEATURES = ['first_listed_date']

assert len(RECONSTRUCT_COLS) == len(set(RECONSTRUCT_COLS)), (
    'Duplicate columns in column parts'
)
assert sorted(COLUMN_ORDER) == sorted(RECONSTRUCT_COLS), (
    'Not all columns accounted for in parts'
)
assert [c in TAB_FEATURES for c in TAB_CAT_FEATURES], (
    'TAB_CAT_FEATURES contains column not in FEATURE_COLS'
)

S3_PREFIX = 'home-listings'