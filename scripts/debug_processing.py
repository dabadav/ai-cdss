# %%

from ai_cdss.data_loader import DataLoader
from ai_cdss.data_processor import DataProcessor


loader = DataLoader(rgs_mode='plus')

nest = [204, 2195, 2913, 2925, 2926, 2937, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3210, 3213, 3222, 3229, 3231, 3318, 3432]
session = loader.load_session_data(nest)
display(session)

processor = DataProcessor()
# %%
# display(session)

##### Effect on demographics on type of content visited (aliisa definition)
##### Heatmap of areas that people visit the most