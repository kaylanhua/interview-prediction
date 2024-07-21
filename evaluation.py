from baseline.tagger import Tagger, ProcessData, Evaluations
import pandas as pd

## GLOBALS
# SEGMENT = 100
FOLDER = './results/'
# SAVE_PATH = f'./results/validation_{SEGMENT}.csv'
SAVE_PATH = './results/validation_full.csv'
checkpoint = True

# LOADING 
validation_set = pd.read_csv('./data/validation.csv')
train_set = pd.read_csv('./data/train_tagged.csv')
train_set = train_set.rename(columns={"Prediction": "zero_shot"})

t_processor = ProcessData(train_set)
train_set = t_processor.split_df()


tagger = Tagger()  
evaluator = Evaluations()

# all_valid = pd.read_csv(SAVE_PATH)

if not checkpoint:
    # ------- STEP ZERO: split input text into question and answer
    v_processor = ProcessData(validation_set)
    validation_set = v_processor.split_df()
    # validation_set = validation_set[:SEGMENT]
    print("\033[92m(0) Finished processing data\033[0m")


    # ------- STEP ONE: do the first round of predictions with the zero-shot classifier (OPENAI)
    df = tagger.apply_function_to_df(validation_set, tagger.simple, include_question=True, col_name="zero_shot")
    # evaluate zero shot classification
    evaluator.evaluate(df, predict_col="zero_shot") 

    # saving zero shot
    df.to_csv('./results/validation_tagged.csv')
    # validation = pd.read_csv('../data/validation_tagged.csv')
    print("\033[92m(1) Finished zero-shot classification\033[0m")


    # ------- STEP TWO: classifying invalid responses
    df = tagger.apply_function_to_df(df, tagger.llm_valid_response, col_name="Valid")
    df.to_csv(FOLDER + 'validation_tagged.csv')

    # df = pd.read_csv(SAVE_PATH)
    df['final_prediction'] = df['Valid'].apply(lambda x: None if x else 1)
    print("\033[92m(2) Finished classifying invalid responses\033[0m")



    # ------- STEP THREE: isolate the valid ones
    all_valid = df[df['Valid'] == True]
    all_valid.to_csv(SAVE_PATH)
    # all_valid = pd.read_csv('./results/validation_valid_all_small.csv')
    print("\033[92m(3) Finished isolating valid responses\033[0m")


# all_valid = pd.read_csv("./results/validation_valid_all.csv")

all_valid = pd.read_csv('./results/claude.csv')

# # ------- STEP 3.5: 4o
# df = tagger.apply_function_to_df(validation_set, tagger.simple, include_question=True, col_name="zero_shot_claude", model="claude")
# # evaluate zero shot classification
# evals = evaluator.evaluate(df, predict_col="zero_shot_claude") 
# print(evals)
# df.to_csv('./results/validation_tagged_claude.csv')

# ------- STEP FOUR: XGBoost
valid_xgb, train_set = tagger.xgb(train_set, all_valid)
print(valid_xgb.head())
# valid_xgb.to_csv(SAVE_PATH)

print("\033[92m(4) Finished XGBoost\033[0m")


# ------- STEP FIVE: add LR and length prediction 
lr_pred, train_set = tagger.length_lr(train_set, valid_xgb)
print(lr_pred.head())
lr_pred.to_csv(SAVE_PATH)


lr_pred = pd.read_csv(SAVE_PATH)

print("\033[92m(5) Finished LR and length prediction\033[0m")


# ------- STEP SIX: add setfit boolean (splitting 1, 2 | 3, 4, 5)


# # ------- STEP SEVEN: add the final predictions
# super_val, train_set = tagger.super_tagger(train_set, lr_pred)

# print(super_val.head())
# super_val.to_csv(SAVE_PATH)

# print("\033[92m(FINAL) Finished super tagger\033[0m")