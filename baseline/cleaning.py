from tagger import Tagger, ProcessData
import pandas as pd

validation_set = pd.read_csv('../data/validation.csv')
processor = ProcessData(validation_set)
validation_set = processor.split_df()

tagger = Tagger()  

# DUMMY PREDICTIONS
df = tagger.apply_function_to_df(validation_set, tagger.simple, include_question=True, col_name="Prediction")
tagger.evaluate(df) # tensor(1.3037)
# df.to_csv('../data/validation_tagged.csv')
validation = pd.read_csv('../data/validation_tagged.csv')

# VALID CHECKING
df = tagger.apply_function_to_df(validation, tagger.llm_valid_response, col_name="Valid")
# df.to_csv('../data/validation_with_valid_classification.csv')
validation_with_valid_classification = pd.read_csv('../data/validation_with_valid_classification.csv')

# KEEPING THE VALID ONES
all_valid = validation[validation['Valid'] == True]
# all_valid.to_csv('../data/validation_valid_all.csv')
all_valid = pd.read_csv('../data/validation_valid_all.csv')